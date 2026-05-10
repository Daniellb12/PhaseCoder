import numpy as np
import scipy.io.wavfile as wav

data = np.load('./tv_integration_test/clip_000003.npz')
audio = data['audio']  # shape (C, 4000)

# Save mic 0 as WAV — listen to it
wav.write('test_speech.wav', 16000, (audio[0] * 32767).astype(np.int16))
print(f"Source: az={float(data['azimuth_deg']):.1f}°, "
      f"dist={float(data['distance_m']):.2f}m, "
      f"dynamic={bool(data['is_dynamic'])}")