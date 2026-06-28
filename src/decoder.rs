use std::fs::File;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// Конвертатор для віспера
pub fn load_audio_data(path: &Path) -> Vec<f32> {
    let file = File::open(path).expect("Не вдалося відкрити аудіофайл");
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        hint.with_extension(&ext.to_string_lossy());
    }

    let fmt_opts: FormatOptions = Default::default();
    let meta_opts: MetadataOptions = Default::default();
    let dec_opts: DecoderOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .expect("Невідомий формат аудіо");
        
    // Дістає Box<dyn FormatReader> зі структури ProbeResult
    let mut format = probed.format;


    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("Не знайдено аудіолінію");

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &dec_opts)
        .expect("Не вдалося створити декодер");

    let mut raw_samples: Vec<f32> = Vec::new();
    let mut sample_rate: u32 = 0;
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break, // кінець файлу
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                if sample_buf.is_none() {
                    let spec = *audio_buf.spec();
                    sample_rate = spec.rate;
                    sample_buf = Some(SampleBuffer::<f32>::new(
                        audio_buf.capacity() as u64,
                        spec,
                    ));
                }

                if let Some(buf) = &mut sample_buf {

                    let channels = audio_buf.spec().channels.count();

                    // Переведення аудіо у mono
                    buf.copy_interleaved_ref(audio_buf);

                    for frame in buf.samples().chunks(channels) {
                        let sum: f32 = frame.iter().sum();
                        raw_samples.push(sum / channels as f32);
                    }
                }
            }
            Err(symphonia::core::errors::Error::IoError(_)) => break,

            Err(e) => {
                eprintln!("Помилка декодування пакета: {}", e);
            }
        }
    }

    // Ресемплінг до 16000 Гц (метод nearest-neighbor)
    if sample_rate != 16000 && sample_rate != 0 {
        let factor = sample_rate as f64 / 16000.0;
        let mut resampled = Vec::new();
        let mut i = 0.0f64;
        while (i as usize) < raw_samples.len() {
            resampled.push(raw_samples[i as usize]);
            i += factor;
        }
        resampled
    } else {
        raw_samples
    }
}