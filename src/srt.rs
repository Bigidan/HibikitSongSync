pub fn align_text_to_segments(
    text_lines: &[&str],
    segments: &[(i64, i64, String)],
) -> Vec<usize> {
    let n = segments.len();
    let m = text_lines.len();

    if m == 0 {
        return vec![0; n];
    }

    // Матриця вартості. чим менше, тим краще збіг
    // cost = 1 - similarity
    let mut cost = vec![vec![1.0f64; m]; n];
    for (i, (_, _, whisper_text)) in segments.iter().enumerate() {
        for (j, line) in text_lines.iter().enumerate() {
            let sim = char_similarity(whisper_text, &line.to_lowercase());
            cost[i][j] = 1.0 - sim;
        }
    }

    // DTW
    let window = ((n.max(m)) / 4).max(3);
    let mut dtw = vec![vec![f64::INFINITY; m]; n];
    dtw[0][0] = cost[0][0];

    for j in 1..m.min(window + 1) {
        dtw[0][j] = dtw[0][j - 1] + cost[0][j];
    }
    for i in 1..n {
        // очікуваний індекс рядка для сегменту "i"
        let expected_j = (i as f64 / n as f64 * m as f64).round() as usize;
        let j_min = expected_j.saturating_sub(window);
        let j_max = (expected_j + window + 1).min(m);

        for j in j_min..j_max {
            let from_diag = if j > 0 { dtw[i - 1][j.saturating_sub(1)] } else { f64::INFINITY };
            let from_left = if j > 0 { dtw[i][j - 1] } else { f64::INFINITY };
            let from_up = dtw[i - 1][j];
            let best = from_diag.min(from_left).min(from_up);
            if best.is_finite() {
                dtw[i][j] = cost[i][j] + best;
            }
        }
    }

    // Зворотній прохід
    let mut mapping = vec![0usize; n];

    let last_expected = (((n - 1) as f64) / n as f64 * m as f64).round() as usize;
    let j_min_last = last_expected.saturating_sub(window);
    let j_max_last = (last_expected + window + 1).min(m);

    let mut j = (j_min_last..j_max_last)
        .filter(|&jj| dtw[n - 1][jj].is_finite())
        .min_by(|&a, &b| dtw[n - 1][a].partial_cmp(&dtw[n - 1][b]).unwrap())
        .unwrap_or(m - 1);

    mapping[n - 1] = j;

    for i in (1..n).rev() {
        let expected_j = (i as f64 / n as f64 * m as f64).round() as usize;
        let j_min = expected_j.saturating_sub(window);

        let diag_j = j.saturating_sub(1);
        let from_diag = if j > 0 && diag_j >= j_min { dtw[i - 1][diag_j] } else { f64::INFINITY };
        let from_left = if j > 0 && j - 1 >= j_min { dtw[i][j - 1] } else { f64::INFINITY };
        let from_up = if j < m { dtw[i - 1][j] } else { f64::INFINITY };

        if from_diag <= from_left && from_diag <= from_up && j > 0 {
            j = diag_j;
        } else if from_up <= from_left {
            // j не змінюється
        } else if j > 0 {
            // залишаємось на j. Кілька сегментів, отже один рядок
        }

        mapping[i - 1] = j.min(m - 1);
    }


    for i in 1..n {
        if mapping[i] < mapping[i - 1] {
            mapping[i] = mapping[i - 1];
        }
    }

    mapping
}

pub fn char_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let bigrams_a: std::collections::HashSet<(char, char)> = a
        .chars()
        .collect::<Vec<_>>()
        .windows(2)
        .map(|w| (w[0], w[1]))
        .collect();

    let bigrams_b: std::collections::HashSet<(char, char)> = b
        .chars()
        .collect::<Vec<_>>()
        .windows(2)
        .map(|w| (w[0], w[1]))
        .collect();

    let intersection = bigrams_a.intersection(&bigrams_b).count();
    let union = bigrams_a.union(&bigrams_b).count();

    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

/// Будує SRT, ГАРАНТОВАНО включаючи кожен рядок тексту рівно один раз.
/// Для рядків, яким не дісталося жодного сегмента (помилка вирівнювання),
/// час підбирається інтерполяцією між сусідніми відомими сегментами,
/// або продовженням від останнього відомого сегмента, якщо це "хвіст" в кінці.
pub fn build_srt(
    text_lines: &[&str],
    segments: &[(i64, i64, String)],
    mapping: &[usize],
) -> String {
    let m = text_lines.len();

    // Для кожного рядка тексту рахуємо мін(t0) і макс(t1) серед сегментів,
    // які на нього вказали.
    let mut line_times: Vec<Option<(i64, i64)>> = vec![None; m];
    for (i, &line_idx) in mapping.iter().enumerate() {
        let (t0, t1) = (segments[i].0, segments[i].1);
        line_times[line_idx] = match line_times[line_idx] {
            Some((cur_t0, cur_t1)) => Some((cur_t0.min(t0), cur_t1.max(t1))),
            None => Some((t0, t1)),
        };
    }

    const DEFAULT_DUR: i64 = 2500; // мс — орієнтовна тривалість для рядків без збігу

    // Заповнюємо "діри" — рядки без жодного сегмента.
    let mut i = 0;
    while i < m {
        if line_times[i].is_some() {
            i += 1;
            continue;
        }

        // Межі порожнього блоку [i..j)
        let mut j = i;
        while j < m && line_times[j].is_none() {
            j += 1;
        }
        let count = j - i;

        let prev_end = if i > 0 { line_times[i - 1].map(|(_, t1)| t1) } else { None };
        let next_start = if j < m { line_times[j].map(|(t0, _)| t0) } else { None };

        match (prev_end, next_start) {
            (Some(p), Some(q)) if q > p => {
                // Є і попередній, і наступний відомий рядок — рівномірно ділимо проміжок
                let span = q - p;
                let step = (span / (count as i64 + 1)).max(1);
                for k in 0..count {
                    let t0 = p + step * (k as i64 + 1);
                    let t1 = (t0 + step).min(q);
                    line_times[i + k] = Some((t0, t1));
                }
            }
            (Some(p), _) => {
                // "Хвіст" в кінці (або наступний час некоректний) — продовжуємо
                // послідовно від останнього відомого сегмента.
                let mut cursor = p;
                for k in 0..count {
                    let t0 = cursor;
                    let t1 = t0 + DEFAULT_DUR;
                    line_times[i + k] = Some((t0, t1));
                    cursor = t1;
                }
            }
            (None, Some(q)) => {
                // Порожньо на самому початку — рахуємо назад від першого відомого
                let mut cursor = q;
                for k in (0..count).rev() {
                    let t1 = cursor;
                    let t0 = (t1 - DEFAULT_DUR).max(0);
                    line_times[i + k] = Some((t0, t1));
                    cursor = t0;
                }
            }
            (None, None) => {
                // Взагалі немає жодного сегмента (напр. Whisper нічого не розпізнав)
                let mut cursor: i64 = 0;
                for k in 0..count {
                    let t0 = cursor;
                    let t1 = t0 + DEFAULT_DUR;
                    line_times[i + k] = Some((t0, t1));
                    cursor = t1;
                }
            }
        }

        i = j;
    }

    // Будуємо фінальний SRT — один запис на кожен рядок тексту, без винятків.
    let mut out = String::new();
    for (idx, line) in text_lines.iter().enumerate() {
        let (t0, t1) = line_times[idx].unwrap_or((0, DEFAULT_DUR));
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            idx + 1,
            format_srt_time(t0),
            format_srt_time(t1),
            line
        ));
    }

    out
}

pub fn format_srt_time(ms: i64) -> String {
    let secs = ms / 1000;
    let msecs = ms % 1000;
    let mins = secs / 60;
    let hours = mins / 60;
    format!("{:02}:{:02}:{:02},{:03}", hours, mins % 60, secs % 60, msecs)
}