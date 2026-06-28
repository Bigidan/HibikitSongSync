use std::env;
use std::path::PathBuf;
use whisper_rs::{FullParams, WhisperContext, WhisperContextParameters};

mod srt;
mod decoder;

use crate::srt::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Якщо користувач не передав 2 файли (аудіо + текст) — зупиняємо програму
    // Працює, якщо просто перетягнути на екзешник два файли
    if args.len() < 3 {
        println!("Помилка: Перетягніть ОДНОЧАСНО аудіофайл та файл тексту на цей ярлик.");
        wait_for_exit();
        return;
    }

    let mut audio_path: Option<PathBuf> = None;
    let mut text_path: Option<PathBuf> = None;

    for arg in &args[1..] {
        let path = PathBuf::from(arg);
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            if ext_str == "mp3" || ext_str == "wav" {
                audio_path = Some(path);
            } else {
                text_path = Some(path);
            }
        }
    }

    match (audio_path, text_path) {
        (Some(audio), Some(text)) => {
            println!("Знайдено аудіо: {:?}", audio.file_name().unwrap());
            println!("Знайдено текст: {:?}", text.file_name().unwrap());
            println!("\nПочинаємо вирівнювання субтитрів...\n");

            let original_text =
                std::fs::read_to_string(&text).expect("Не вдалося прочитати текст");

            let text_lines: Vec<&str> = original_text
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .collect();

            let model_path = std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .join("models/ggml-base.bin");

            let ctx = WhisperContext::new_with_params(
                model_path.to_str().unwrap(),
                WhisperContextParameters::default(),
            )
            .expect("Не вдалося завантажити модель");

            let mut state = ctx.create_state().expect("Не вдалося створити стан");

            let mut params =
                FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });

            params.set_language(Some("uk"));
            params.set_initial_prompt(&original_text);
            params.set_print_timestamps(false);
            params.set_suppress_non_speech_tokens(true);
            params.set_logprob_thold(-2.0);
            params.set_no_speech_thold(0.8);

            let audio_data = decoder::load_audio_data(&audio);
            state.full(params, &audio_data).expect("Помилка транскрибації");

            let num_segments = state
                .full_n_segments()
                .expect("Не вдалося отримати кількість сегментів") as usize;



            
            println!("═══════════════════════════════════════════════════════");
            println!("  WHISPER ЗНАЙШОВ {} СЕГМЕНТІВ:", num_segments);
            println!("═══════════════════════════════════════════════════════");

            let mut segments: Vec<(i64, i64, String)> = Vec::new();

            for i in 0..num_segments {
                let t0 = state.full_get_segment_t0(i as i32).unwrap_or(0) * 10;
                let t1 = state.full_get_segment_t1(i as i32).unwrap_or(0) * 10;
                let whisper_text = state
                    .full_get_segment_text(i as i32)
                    .unwrap_or_default()
                    .trim()
                    .to_string();

                println!(
                    "  [{:>3}] {} --> {}  | \"{}\"",
                    i + 1,
                    format_srt_time(t0),
                    format_srt_time(t1),
                    whisper_text
                );

                segments.push((t0, t1, whisper_text.to_lowercase()));
            }

            println!("═══════════════════════════════════════════════════════");




            println!("\n  ПАУЗИ між сегментами (> 0.5 сек):");
            for i in 1..segments.len() {
                let gap_ms = segments[i].0 - segments[i - 1].1;
                if gap_ms > 500 {
                    println!(
                        "    після сегм. {}: пауза {:.1} сек ({} --> {})",
                        i,
                        gap_ms as f64 / 1000.0,
                        format_srt_time(segments[i - 1].1),
                        format_srt_time(segments[i].0)
                    );
                }
            }

            println!("\n  У тексті {} рядків:", text_lines.len());
            for (i, line) in text_lines.iter().enumerate() {
                println!("  [{:>3}] {}", i + 1, line);
            }
            println!("═══════════════════════════════════════════════════════\n");

            let srt_content = if num_segments == 0 {
                println!("Whisper не знайшов сегментів, записуємо текст без тімінгів...");
                let mut out = String::new();
                for (i, line) in text_lines.iter().enumerate() {
                    out.push_str(&format!(
                        "{}\n00:00:00,000 --> 00:00:00,000\n{}\n\n",
                        i + 1,
                        line
                    ));
                }
                out
            } else {
                let mapping = align_text_to_segments(&text_lines, &segments);



                println!("  РЕЗУЛЬТАТ ВИРІВНЮВАННЯ (сегмент → рядок тексту):");
                for (i, &line_idx) in mapping.iter().enumerate() {
                    println!(
                        "  сегм. {:>3} ({} --> {})  →  рядок {:>3}: \"{}\"",
                        i + 1,
                        format_srt_time(segments[i].0),
                        format_srt_time(segments[i].1),
                        line_idx + 1,
                        text_lines.get(line_idx).unwrap_or(&"???")
                    );
                }
                println!();

                build_srt(&text_lines, &segments, &mapping)
            };

            let mut srt_path = text.clone();
            srt_path.set_extension("srt");
            std::fs::write(&srt_path, &srt_content).expect("Не вдалося зберегти SRT");

            println!("Готово! Файл субтитрів: {:?}", srt_path);
        }
        _ => {
            println!("Помилка: Не знайдено пару аудіо + текст.");
        }
    }

    wait_for_exit();
}


fn wait_for_exit() {
    println!("\nНатисніть Enter для виходу...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
}