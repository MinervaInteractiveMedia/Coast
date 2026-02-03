# COAST
### Parametric coastal sound synthesizer

Coast is a small program the synthesizes coatal sounds parametrically. It does so with a combination of fft synthesis and noise modulation.

---

## Quick Start

```bash
# Required
pip install numpy pillow

# Optional (for playback)
pip install sounddevice
```

---

## What's Inside

| File | Role |
|---|---|
| `coast.py` | The main python script |
| `coast.app` | MACOS executable |


---

##Features:
🌊 Ocean Waves

Volume, frequency, intensity, and foam/splash controls
Realistic wave patterns with rumble and splash components

💨 Wind

Volume, speed, gust frequency, and gust intensity
Dynamic wind with realistic gusting patterns

🐦 Seabirds

Volume, call frequency, and species variety (5 different bird types)
Seagulls, chirps, caws, whistles, and complex calls

🌧️ Rain

Volume, intensity, and droplet size
Light rain with distinct droplets or heavy rain texture

⚡ Thunder

Volume, frequency, and distance controls
Close cracks or distant rumbles

⚙️ Export Options

Sample rates: 22050, 44100, 48000, 96000 Hz
Bit depths: 16, 24, 32 bit
Duration: 5-300 seconds
High-quality WAV export

---

## How to Use

### 1. Adjust sliders to create your desired coastal mood


### 2. Click "Generate Sound"

### 3. Play it back (if sounddevice installed) or export directly

### 4. Export as WAV with your chosen quality settings
