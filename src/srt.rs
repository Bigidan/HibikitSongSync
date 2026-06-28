
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

    // Йобані біграми
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

    // Jaccard similarity
    let intersection = bigrams_a.intersection(&bigrams_b).count();
    let union = bigrams_a.union(&bigrams_b).count();

    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

pub fn build_srt(
    text_lines: &[&str],
    segments: &[(i64, i64, String)],
    mapping: &[usize],
) -> String {
    let mut out = String::new();
    let mut srt_index = 1;
    let mut i = 0;

    // Групування сегментів блоками
    while i < segments.len() {
        let line_idx = mapping[i];
        let t0 = segments[i].0;
        
        let mut j = i;
        while j + 1 < segments.len() && mapping[j + 1] == line_idx {
            j += 1;
        }
        let t1 = segments[j].1;

        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            srt_index,
            format_srt_time(t0),
            format_srt_time(t1),
            text_lines[line_idx]
        ));

        srt_index += 1;
        i = j + 1;
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