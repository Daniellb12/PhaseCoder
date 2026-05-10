import numpy as np
import scipy.io.wavfile as wav
from librispeech_source import LibriSpeechProvider

provider = LibriSpeechProvider(
    cache_dir="./librispeech_cache",
    subset="dev-clean",
    max_utterances=2000,
    synthetic_fraction=0.1,
    verbose=True,
)

# Generate 10 source signals and concatenate them with gaps
rng = np.random.default_rng(42)
combined = []
for i in range(10):
    sig = provider.get_signal(duration_s=2.0, rng=rng)  # 2 seconds each
    combined.append(sig)
    combined.append(np.zeros(int(0.5 * 16000), dtype=np.float32))  # 500ms gap
    
    # Print stats so we know what's in each one
    peak = np.abs(sig).max()
    rms = np.sqrt(np.mean(sig**2))
    print(f"Signal {i}: peak={peak:.3f}, rms={rms:.3f}, "
          f"len={len(sig)/16000:.2f}s, "
          f"likely_synthetic={'YES' if rms < 0.15 and peak < 0.85 else 'maybe real speech'}")

combined = np.concatenate(combined)
wav.write('source_only_test.wav', 16000, (combined * 32767).clip(-32768, 32767).astype(np.int16))
print(f"\nWrote source_only_test.wav: {len(combined)/16000:.1f} seconds")