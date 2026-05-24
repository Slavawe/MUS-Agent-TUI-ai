use std::process::Command;
use std::time::Instant;

/// Result of a TTS or recording operation
#[derive(Debug)]
pub struct VoxResult {
    pub wav_path: Option<String>,
    pub duration_ms: u64,
    pub sample_rate: u32,
    pub text: String,
}

/// Voice modulation parameters derived from neurochemistry
#[derive(Debug, Clone, Copy)]
pub struct VoiceModulation {
    pub speed: f32,
    pub pitch: f32,
    pub volume: f32,
    pub emphasis: f32,
}

impl VoiceModulation {
    pub fn from_neurochem(dopamine: f32, adrenaline: f32) -> Self {
        let speed = 1.0 + (dopamine * 0.08) - (adrenaline * 0.04);
        let pitch = 1.0 + (dopamine * 0.06) - (adrenaline * 0.08);
        let volume = (0.5 + (dopamine * 0.2) + (adrenaline * 0.1)).min(1.0);
        let emphasis = (dopamine * 0.3 + adrenaline * 0.5).min(1.0);
        VoiceModulation { speed, pitch, volume, emphasis }
    }
}

/// Available TTS backends
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TtsBackend {
    Piper,
    EspeakNg,
    None,
}

impl TtsBackend {
    pub fn detect() -> Self {
        if Self::check_piper() { TtsBackend::Piper }
        else if Self::check_espeak() { TtsBackend::EspeakNg }
        else { TtsBackend::None }
    }

    fn check_piper() -> bool {
        which::which("piper").is_ok()
    }

    fn check_espeak() -> bool {
        which::which("espeak-ng").is_ok() || which::which("espeak").is_ok()
    }

    pub fn name(&self) -> &'static str {
        match self {
            TtsBackend::Piper => "Piper",
            TtsBackend::EspeakNg => "eSpeak-NG",
            TtsBackend::None => "none",
        }
    }
}

/// MUS Vox engine — local TTS with neurochemical modulation
pub struct MusVoxEngine {
    pub backend: TtsBackend,
    pub voice_fingerprint_path: Option<String>,
    pub current_language: String,
    pub output_dir: String,
    pub enabled: bool,
    pub last_result: Option<VoxResult>,
}

impl MusVoxEngine {
    pub fn new() -> Self {
        let backend = TtsBackend::detect();
        let out_dir = "/tmp/mus_vox".to_string();
        let _ = std::fs::create_dir_all(&out_dir);
        MusVoxEngine {
            backend,
            voice_fingerprint_path: None,
            current_language: "ru".to_string(),
            output_dir: out_dir,
            enabled: backend != TtsBackend::None,
            last_result: None,
        }
    }

    pub fn is_available(&self) -> bool {
        self.backend != TtsBackend::None
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Speak text with neurochemical modulation
    pub fn speak(&mut self, text: &str, dopamine: f32, adrenaline: f32) -> Option<VoxResult> {
        if !self.enabled { return None; }
        let modul = VoiceModulation::from_neurochem(dopamine, adrenaline);
        let out_path = format!("{}/out_{}.wav", self.output_dir, Instant::now().elapsed().as_nanos() % 100000);
        let start = Instant::now();

        let result = match self.backend {
            TtsBackend::Piper => self.speak_piper(text, &modul, &out_path),
            TtsBackend::EspeakNg => self.speak_espeak(text, &modul, &out_path),
            TtsBackend::None => {
                let freq = 200.0 + dopamine * 50.0 + adrenaline * 30.0;
                let dur = 0.3 + (text.len() as f32 * 0.02).min(2.0);
                Self::generate_test_tone(&out_path, freq, dur);
                Some(VoxResult {
                    wav_path: Some(out_path.clone()),
                    duration_ms: 0,
                    sample_rate: 22050,
                    text: text.to_string(),
                })
            }
        };

        if let Some(r) = &result {
            self.last_result = Some(VoxResult {
                wav_path: Some(out_path.clone()),
                duration_ms: start.elapsed().as_millis() as u64,
                sample_rate: r.sample_rate,
                text: text.to_string(),
            });
        }
        result
    }

    fn speak_piper(&self, text: &str, modul: &VoiceModulation, out_path: &str) -> Option<VoxResult> {
        let model = self.voice_fingerprint_path.as_deref().unwrap_or("voice.onnx");
        let length_scale = format!("{:.2}", (1.0 / modul.speed).max(0.5).min(2.0));
        let mut child = Command::new("piper")
            .arg("--model").arg(model)
            .arg("--output-file").arg(out_path)
            .arg("--length-scale").arg(&length_scale)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn().ok()?;

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            let _ = stdin.write_all(text.as_bytes());
        }
        let _ = child.wait();
        Some(VoxResult {
            wav_path: Some(out_path.to_string()),
            duration_ms: 0,
            sample_rate: 22050,
            text: text.to_string(),
        })
    }

    fn speak_espeak(&self, text: &str, modul: &VoiceModulation, out_path: &str) -> Option<VoxResult> {
        let espeak = if which::which("espeak-ng").is_ok() { "espeak-ng" } else { "espeak" };
        let speed = (modul.speed * 175.0) as i32;
        let pitch = (modul.pitch * 50.0) as i32;
        let amp = (modul.volume * 200.0) as i32;

        let out = Command::new(espeak)
            .arg("-w").arg(out_path)
            .arg("-s").arg(speed.to_string())
            .arg("-p").arg(pitch.to_string())
            .arg("-a").arg(amp.to_string())
            .arg("-v").arg(&self.current_language)
            .arg(text)
            .output().ok()?;

        if out.status.success() {
            Some(VoxResult {
                wav_path: Some(out_path.to_string()),
                duration_ms: 0,
                sample_rate: 22050,
                text: text.to_string(),
            })
        } else {
            None
        }
    }

    /// Play a WAV file via aplay
    pub fn play_wav(path: &str) -> bool {
        Command::new("aplay")
            .arg(path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Record voice sample via arecord
    pub fn record_sample(duration_secs: u32, out_path: &str) -> bool {
        Command::new("arecord")
            .arg("-d").arg(duration_secs.to_string())
            .arg("-f").arg("cd")
            .arg("-t").arg("wav")
            .arg(out_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Generate minimal test tone (no TTS needed) — for testing the pipeline
    pub fn generate_test_tone(path: &str, frequency: f32, duration_secs: f32) -> bool {
        let sample_rate = 22050u32;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let mut wav_data = Vec::with_capacity(num_samples * 2);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (t * frequency * 2.0 * std::f32::consts::PI).sin();
            let amplitude = (1.0 - (t / duration_secs)).max(0.0);
            let sample_i16 = (sample * amplitude * 0.3 * 32767.0) as i16;
            wav_data.extend_from_slice(&sample_i16.to_le_bytes());
        }

        let data_size = wav_data.len() as u32;
        let file_size = 36 + data_size;
        let sample_rate_le = sample_rate.to_le_bytes();
        let byte_rate = (sample_rate * 2).to_le_bytes();
        let data_size_le = data_size.to_le_bytes();
        let file_size_le = file_size.to_le_bytes();

        let header: Vec<u8> = [
            b'R', b'I', b'F', b'F',
            file_size_le[0], file_size_le[1], file_size_le[2], file_size_le[3],
            b'W', b'A', b'V', b'E',
            b'f', b'm', b't', b' ',
            16, 0, 0, 0,  // chunk size = 16
            1, 0,         // PCM
            1, 0,         // mono
            sample_rate_le[0], sample_rate_le[1], sample_rate_le[2], sample_rate_le[3],
            byte_rate[0], byte_rate[1], byte_rate[2], byte_rate[3],
            2, 0,         // block align
            16, 0,        // bits per sample
            b'd', b'a', b't', b'a',
            data_size_le[0], data_size_le[1], data_size_le[2], data_size_le[3],
        ].to_vec();

        let mut wav = header;
        wav.extend_from_slice(&wav_data);
        std::fs::write(path, &wav).is_ok()
    }
}

impl Default for MusVoxEngine {
    fn default() -> Self { Self::new() }
}
