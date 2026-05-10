import numpy as np
import scipy.io.wavfile as wav
import os

clip_dir = './tv_train_data'
clips = sorted([f for f in os.listdir(clip_dir) if f.endswith('.npz')])

# Build a longer concatenation and AMPLIFY it
combined = []
for clip_name in clips[:10]:
    data = np.load(f'{clip_dir}/{clip_name}')
    audio = data['audio'][0]  # mic 0
    
    peak = np.abs(audio).max()
    rms = np.sqrt(np.mean(audio**2))
    print(f"{clip_name}: peak={peak:.3f}, rms={rms:.3f}, "
          f"is_dynamic={bool(data['is_dynamic'])}")
    
    # Amplify by 4x to see if there's just signal that's too quiet
    amplified = np.clip(audio * 4.0, -1.0, 1.0)
    combined.append(amplified)
    combined.append(np.zeros(int(0.5 * 16000), dtype=np.float32))

combined = np.concatenate(combined)
wav.write('amplified_clips.wav', 16000, (combined * 32767).clip(-32768, 32767).astype(np.int16))