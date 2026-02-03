#!/usr/bin/env python3
"""
Coastal Sound Synthesizer
A GUI application for creating and mixing coastal soundscapes with customizable parameters.
Supports WAV export with selectable sample rates and bit depths.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import wave
import struct
import threading
from pathlib import Path
import io


class CoastalSoundSynthesizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Coastal Sound Synthesizer")
        self.root.geometry("800x900")
        
        # Audio parameters
        self.sample_rate = 44100
        self.bit_depth = 16
        self.duration = 30  # seconds
        
        # Playback state
        self.is_playing = False
        self.audio_data = None
        self.playback_thread = None
        
        # Try to import sounddevice for playback
        self.sounddevice_available = False
        try:
            import sounddevice as sd
            self.sd = sd
            self.sounddevice_available = True
        except ImportError:
            pass
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container with scrollbar
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="Coastal Sound Synthesizer", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(pady=15)
        
        # ===== WAVES SECTION =====
        waves_frame = ttk.LabelFrame(scrollable_frame, text="🌊 Ocean Waves", padding="15")
        waves_frame.pack(fill="x", padx=20, pady=10)
        
        self.wave_volume = self._create_slider(waves_frame, "Volume", 0, 100, 70)
        self.wave_frequency = self._create_slider(waves_frame, "Wave Frequency (waves/min)", 5, 40, 15)
        self.wave_intensity = self._create_slider(waves_frame, "Intensity", 0, 100, 60)
        self.wave_foam = self._create_slider(waves_frame, "Foam/Splash", 0, 100, 40)
        
        # ===== WIND SECTION =====
        wind_frame = ttk.LabelFrame(scrollable_frame, text="💨 Wind", padding="15")
        wind_frame.pack(fill="x", padx=20, pady=10)
        
        self.wind_volume = self._create_slider(wind_frame, "Volume", 0, 100, 50)
        self.wind_speed = self._create_slider(wind_frame, "Speed", 0, 100, 40)
        self.wind_gust_freq = self._create_slider(wind_frame, "Gust Frequency", 0, 100, 30)
        self.wind_gust_intensity = self._create_slider(wind_frame, "Gust Intensity", 0, 100, 50)
        
        # ===== BIRDS SECTION =====
        birds_frame = ttk.LabelFrame(scrollable_frame, text="🐦 Seabirds", padding="15")
        birds_frame.pack(fill="x", padx=20, pady=10)
        
        self.bird_volume = self._create_slider(birds_frame, "Volume", 0, 100, 30)
        self.bird_frequency = self._create_slider(birds_frame, "Call Frequency", 0, 100, 25)
        self.bird_variety = self._create_slider(birds_frame, "Species Variety", 1, 5, 3)
        
        # ===== RAIN SECTION =====
        rain_frame = ttk.LabelFrame(scrollable_frame, text="🌧️ Rain", padding="15")
        rain_frame.pack(fill="x", padx=20, pady=10)
        
        self.rain_volume = self._create_slider(rain_frame, "Volume", 0, 100, 0)
        self.rain_intensity = self._create_slider(rain_frame, "Intensity", 0, 100, 50)
        self.rain_droplet_size = self._create_slider(rain_frame, "Droplet Size", 0, 100, 50)
        
        # ===== THUNDER SECTION =====
        thunder_frame = ttk.LabelFrame(scrollable_frame, text="⚡ Thunder", padding="15")
        thunder_frame.pack(fill="x", padx=20, pady=10)
        
        self.thunder_volume = self._create_slider(thunder_frame, "Volume", 0, 100, 0)
        self.thunder_frequency = self._create_slider(thunder_frame, "Frequency", 0, 100, 10)
        self.thunder_distance = self._create_slider(thunder_frame, "Distance", 0, 100, 50)
        
        # ===== GLOBAL SETTINGS =====
        settings_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Global Settings", padding="15")
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        # Duration
        duration_frame = ttk.Frame(settings_frame)
        duration_frame.pack(fill="x", pady=5)
        ttk.Label(duration_frame, text="Duration (seconds):").pack(side="left", padx=5)
        self.duration_var = tk.IntVar(value=30)
        duration_spinner = ttk.Spinbox(duration_frame, from_=5, to=300, 
                                      textvariable=self.duration_var, width=10)
        duration_spinner.pack(side="left", padx=5)
        
        # Sample Rate
        sr_frame = ttk.Frame(settings_frame)
        sr_frame.pack(fill="x", pady=5)
        ttk.Label(sr_frame, text="Sample Rate:").pack(side="left", padx=5)
        self.sample_rate_var = tk.StringVar(value="44100")
        sr_combo = ttk.Combobox(sr_frame, textvariable=self.sample_rate_var,
                               values=["22050", "44100", "48000", "96000"], 
                               state="readonly", width=12)
        sr_combo.pack(side="left", padx=5)
        ttk.Label(sr_frame, text="Hz").pack(side="left")
        
        # Bit Depth
        bd_frame = ttk.Frame(settings_frame)
        bd_frame.pack(fill="x", pady=5)
        ttk.Label(bd_frame, text="Bit Depth:").pack(side="left", padx=5)
        self.bit_depth_var = tk.StringVar(value="16")
        bd_combo = ttk.Combobox(bd_frame, textvariable=self.bit_depth_var,
                               values=["16", "24", "32"], 
                               state="readonly", width=12)
        bd_combo.pack(side="left", padx=5)
        ttk.Label(bd_frame, text="bit").pack(side="left")
        
        # ===== CONTROL BUTTONS =====
        control_frame = ttk.Frame(scrollable_frame, padding="15")
        control_frame.pack(fill="x", padx=20, pady=10)
        
        self.generate_btn = ttk.Button(control_frame, text="🎵 Generate Sound", 
                                      command=self.generate_sound)
        self.generate_btn.pack(side="left", padx=5)
        
        self.play_btn = ttk.Button(control_frame, text="▶️ Play", 
                                  command=self.play_sound, state=tk.DISABLED)
        self.play_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop", 
                                  command=self.stop_sound, state=tk.DISABLED)
        self.stop_btn.pack(side="left", padx=5)
        
        self.export_btn = ttk.Button(control_frame, text="💾 Export WAV", 
                                    command=self.export_wav, state=tk.DISABLED)
        self.export_btn.pack(side="left", padx=5)
        
        # Status
        self.status_label = ttk.Label(scrollable_frame, text="Ready", 
                                     foreground="blue", font=('Arial', 10))
        self.status_label.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(scrollable_frame, mode='indeterminate')
        self.progress.pack(fill="x", padx=20, pady=5)
        
        if not self.sounddevice_available:
            info_label = ttk.Label(scrollable_frame, 
                                  text="Note: Install 'sounddevice' for audio playback\n(pip install sounddevice)",
                                  foreground="orange", font=('Arial', 9))
            info_label.pack(pady=5)
    
    def _create_slider(self, parent, label, min_val, max_val, default):
        """Create a labeled slider with value display"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        
        label_widget = ttk.Label(frame, text=label, width=25, anchor="w")
        label_widget.pack(side="left", padx=5)
        
        var = tk.DoubleVar(value=default)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, 
                          variable=var, orient="horizontal")
        slider.pack(side="left", fill="x", expand=True, padx=5)
        
        value_label = ttk.Label(frame, text=f"{default:.1f}", width=6)
        value_label.pack(side="left", padx=5)
        
        # Update value label when slider moves
        def update_label(*args):
            value_label.config(text=f"{var.get():.1f}")
        var.trace_add('write', update_label)
        
        return var
    
    def generate_sound(self):
        """Generate the coastal soundscape"""
        self.generate_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Generating soundscape...", foreground="blue")
        
        thread = threading.Thread(target=self._generate_audio)
        thread.start()
    
    def _generate_audio(self):
        """Generate audio in background thread"""
        try:
            # Get parameters
            duration = self.duration_var.get()
            sample_rate = int(self.sample_rate_var.get())
            
            # Generate time array
            t = np.linspace(0, duration, int(duration * sample_rate))
            
            # Initialize audio
            audio = np.zeros(len(t))
            
            # Generate each sound layer
            self.root.after(0, lambda: self.status_label.config(text="Generating waves..."))
            audio += self._generate_waves(t, sample_rate)
            
            self.root.after(0, lambda: self.status_label.config(text="Generating wind..."))
            audio += self._generate_wind(t, sample_rate)
            
            self.root.after(0, lambda: self.status_label.config(text="Generating birds..."))
            audio += self._generate_birds(t, sample_rate)
            
            self.root.after(0, lambda: self.status_label.config(text="Generating rain..."))
            audio += self._generate_rain(t, sample_rate)
            
            self.root.after(0, lambda: self.status_label.config(text="Generating thunder..."))
            audio += self._generate_thunder(t, sample_rate)
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9
            
            self.audio_data = audio
            self.sample_rate = sample_rate
            
            self.root.after(0, self._generation_complete)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", 
                                                           f"Failed to generate sound: {str(e)}"))
            self.root.after(0, self._reset_ui)
    
    def _generate_waves(self, t, sample_rate):
        """Generate ocean wave sounds"""
        volume = self.wave_volume.get() / 100.0
        if volume == 0:
            return np.zeros(len(t))
        
        frequency = self.wave_frequency.get() / 60.0  # Convert to Hz
        intensity = self.wave_intensity.get() / 100.0
        foam = self.wave_foam.get() / 100.0
        
        # Create continuous base rumble (always present)
        continuous_rumble = self._filtered_noise(len(t), sample_rate, 20, 150)
        continuous_rumble *= 0.3  # Keep it subtle
        
        # Create wave crashes with overlapping envelopes
        wave_audio = np.zeros(len(t))
        duration = len(t) / sample_rate
        
        # Calculate wave period
        wave_period = 1.0 / frequency if frequency > 0 else 4.0
        num_waves = int(duration / wave_period) + 2
        
        for i in range(num_waves):
            # Wave timing with some randomness
            wave_time = i * wave_period + np.random.uniform(-0.2, 0.2)
            if wave_time < 0 or wave_time >= duration:
                continue
            
            wave_start = int(wave_time * sample_rate)
            
            # Wave duration varies
            wave_duration = wave_period * (1.5 + intensity * 1.5)
            wave_samples = int(wave_duration * sample_rate)
            
            # Clip to available space
            if wave_start + wave_samples > len(t):
                wave_samples = len(t) - wave_start
            
            if wave_samples <= 0:
                continue
            
            # Create wave envelope (rise, peak, decay)
            attack_time = wave_duration * 0.2
            
            envelope = np.zeros(wave_samples)
            attack_samples = int(attack_time * sample_rate)
            
            # Ensure attack_samples doesn't exceed wave_samples
            attack_samples = min(attack_samples, wave_samples)
            
            # Attack phase (wave builds)
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 2
            
            # Decay phase (wave recedes)
            if attack_samples < wave_samples:
                decay_samples = wave_samples - attack_samples
                envelope[attack_samples:] = np.exp(-np.linspace(0, 3 + intensity * 2, decay_samples))
            
            # Generate wave components
            # Low rumble
            rumble_freq = 50 + np.random.uniform(-10, 10)
            rumble = self._filtered_noise(wave_samples, sample_rate, 30, 200)
            rumble *= envelope
            
            # Mid-range wash
            wash = self._filtered_noise(wave_samples, sample_rate, 200, 1500)
            wash *= envelope * 0.5
            
            # High frequency splash/foam
            splash = self._filtered_noise(wave_samples, sample_rate, 1500, 6000)
            splash *= envelope * foam * 0.7
            
            # Add some sizzle at the peak
            if foam > 0.3:
                peak_pos = attack_samples
                sizzle_duration = int(0.3 * sample_rate)
                if peak_pos + sizzle_duration < wave_samples:
                    sizzle = self._filtered_noise(sizzle_duration, sample_rate, 4000, 8000)
                    sizzle_env = np.exp(-np.linspace(0, 5, sizzle_duration))
                    sizzle *= sizzle_env * foam
                    splash[peak_pos:peak_pos + sizzle_duration] += sizzle
            
            # Combine wave components
            single_wave = rumble * 0.6 + wash * 0.3 + splash * 0.3
            
            # Apply smooth fade in/out to prevent clicks (first/last 5ms)
            fade_samples = min(int(0.005 * sample_rate), wave_samples // 10)
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                single_wave[:fade_samples] *= fade_in
                single_wave[-fade_samples:] *= fade_out
            
            # Vary intensity
            wave_intensity_variation = 0.7 + np.random.uniform(0, 0.3) * intensity
            single_wave *= wave_intensity_variation
            
            # Add to main audio
            wave_audio[wave_start:wave_start + wave_samples] += single_wave
        
        # Combine continuous rumble with wave crashes
        waves = continuous_rumble + wave_audio * 0.8
        
        return waves * volume
    
    def _generate_wind(self, t, sample_rate):
        """Generate wind sounds"""
        volume = self.wind_volume.get() / 100.0
        if volume == 0:
            return np.zeros(len(t))
        
        speed = self.wind_speed.get() / 100.0
        gust_freq = self.wind_gust_freq.get() / 100.0
        gust_intensity = self.wind_gust_intensity.get() / 100.0
        
        # Base wind noise (low to mid frequency)
        base_wind = self._filtered_noise(len(t), sample_rate, 100, 2000 + speed * 3000)
        
        # Gusts (slow modulation)
        gust_lfo = np.sin(2 * np.pi * gust_freq * 0.1 * t) * gust_intensity
        gust_lfo = (gust_lfo + 1) / 2  # Normalize to 0-1
        gust_lfo = np.power(gust_lfo, 2)  # Sharper gusts
        
        # Apply gust modulation
        wind = base_wind * (0.5 + 0.5 * speed + 0.5 * gust_lfo)
        
        return wind * volume
    
    def _generate_birds(self, t, sample_rate):
        """Generate seabird calls"""
        volume = self.bird_volume.get() / 100.0
        if volume == 0:
            return np.zeros(len(t))
        
        frequency = self.bird_frequency.get() / 100.0
        variety = int(self.bird_variety.get())
        
        birds = np.zeros(len(t))
        duration = len(t) / sample_rate
        
        # Calculate number of bird calls
        num_calls = int(frequency * duration * 5)  # Scale factor for reasonable amount
        
        for _ in range(num_calls):
            # Random timing
            call_duration_max = 1.5  # Max call duration
            if duration < call_duration_max + 0.1:
                continue
            start_time = np.random.uniform(0, duration - call_duration_max)
            start_sample = int(start_time * sample_rate)
            
            if start_sample >= len(birds):
                continue
            
            # Random bird type
            bird_type = np.random.randint(0, variety)
            
            # Generate call
            call = self._generate_bird_call(sample_rate, bird_type)
            
            # Add to mix with bounds checking
            end_sample = min(start_sample + len(call), len(birds))
            birds[start_sample:end_sample] += call[:end_sample - start_sample]
        
        return birds * volume * 0.5  # Scale down to avoid clipping
    
    def _generate_bird_call(self, sample_rate, bird_type):
        """Generate a single bird call"""
        duration = np.random.uniform(0.3, 1.5)
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        if bird_type == 0:  # Seagull
            freq = np.linspace(2000, 1200, len(t))
            call = np.sin(2 * np.pi * freq * t / sample_rate * 50)
            envelope = np.exp(-t * 3) * (1 - np.exp(-t * 20))
        elif bird_type == 1:  # High pitched chirp
            freq = 3000 + 500 * np.sin(2 * np.pi * 15 * t)
            call = np.sin(2 * np.pi * freq * t / sample_rate * 50)
            envelope = np.exp(-t * 5)
        elif bird_type == 2:  # Low caw
            freq = 800 + 200 * np.sin(2 * np.pi * 5 * t)
            call = np.sin(2 * np.pi * freq * t / sample_rate * 50)
            envelope = np.exp(-t * 2)
        elif bird_type == 3:  # Whistle
            freq = 2500
            call = np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-t * 4) * np.sin(np.pi * t / duration)
        else:  # Complex call
            call = np.sin(2 * np.pi * 1500 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)
            envelope = np.exp(-t * 3) * (1 + 0.5 * np.sin(2 * np.pi * 8 * t))
        
        return call * envelope
    
    def _generate_rain(self, t, sample_rate):
        """Generate rain sounds"""
        volume = self.rain_volume.get() / 100.0
        if volume == 0:
            return np.zeros(len(t))
        
        intensity = self.rain_intensity.get() / 100.0
        droplet_size = self.rain_droplet_size.get() / 100.0
        
        # High frequency noise for rain texture
        rain_texture = self._filtered_noise(len(t), sample_rate, 
                                           2000 - droplet_size * 1000, 
                                           8000 + droplet_size * 4000)
        
        # Intensity modulation (more rain = more consistent)
        intensity_lfo = 0.8 + 0.2 * intensity + 0.2 * (1 - intensity) * np.random.random(len(t))
        
        # Add individual droplet impacts
        if intensity < 0.5:  # Light rain - distinct droplets
            num_drops = int(intensity * len(t) * 0.01)
            for _ in range(num_drops):
                drop_duration = int(sample_rate * 0.02 * (1 + droplet_size))
                if drop_duration >= len(t) - 1:
                    continue
                drop_pos = np.random.randint(0, len(t) - drop_duration)
                drop_t = np.linspace(0, 1, drop_duration)
                drop = np.sin(2 * np.pi * (4000 + droplet_size * 2000) * drop_t) * np.exp(-drop_t * 20)
                # Ensure we don't exceed array bounds
                end_pos = min(drop_pos + drop_duration, len(t))
                actual_drop_len = end_pos - drop_pos
                rain_texture[drop_pos:end_pos] += drop[:actual_drop_len]
        
        rain = rain_texture * intensity_lfo * intensity
        
        return rain * volume
    
    def _generate_thunder(self, t, sample_rate):
        """Generate thunder sounds"""
        volume = self.thunder_volume.get() / 100.0
        if volume == 0:
            return np.zeros(len(t))
        
        frequency = self.thunder_frequency.get() / 100.0
        distance = self.thunder_distance.get() / 100.0
        
        thunder = np.zeros(len(t))
        duration = len(t) / sample_rate
        
        # Calculate number of thunder events
        num_strikes = int(frequency * duration * 2)
        
        for _ in range(num_strikes):
            # Random timing
            start_time = np.random.uniform(0, duration - 3)
            start_sample = int(start_time * sample_rate)
            
            # Generate thunder strike
            strike = self._generate_thunder_strike(sample_rate, distance)
            
            # Add to mix
            end_sample = min(start_sample + len(strike), len(thunder))
            thunder[start_sample:end_sample] += strike[:end_sample - start_sample]
        
        return thunder * volume * 0.7
    
    def _generate_thunder_strike(self, sample_rate, distance):
        """Generate a single thunder strike"""
        # Duration increases with distance (reverb/rumble)
        duration = 2 + distance * 3
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Initial crack (close) or rumble (far)
        if distance < 0.5:
            # Sharp crack
            crack_duration = 0.1
            crack_samples = int(crack_duration * sample_rate)
            crack = self._filtered_noise(crack_samples, sample_rate, 1000, 8000)
            crack_env = np.exp(-np.linspace(0, 30, crack_samples))
            crack *= crack_env
            
            # Rumble
            rumble_samples = len(t) - crack_samples
            rumble_t = np.linspace(0, duration - crack_duration, rumble_samples)
            rumble = self._filtered_noise(rumble_samples, sample_rate, 20, 200)
            rumble *= np.exp(-rumble_t * (2 - distance))
            
            # Crossfade between crack and rumble to prevent clicking
            crossfade_samples = min(int(0.02 * sample_rate), crack_samples // 2, rumble_samples // 2)
            if crossfade_samples > 0:
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                crack[-crossfade_samples:] *= fade_out
                rumble[:crossfade_samples] *= fade_in
            
            strike = np.concatenate([crack, rumble])
        else:
            # Distant rumble only
            strike = self._filtered_noise(len(t), sample_rate, 30, 300)
            strike *= np.exp(-t * 0.8)
        
        # Add some randomness to the envelope
        envelope_variation = 1 + 0.3 * np.sin(2 * np.pi * np.random.uniform(2, 8) * np.linspace(0, duration, len(strike)))
        strike *= envelope_variation
        
        # Smooth fade out at the end to prevent clicks
        fade_samples = min(int(0.01 * sample_rate), len(strike) // 10)
        if fade_samples > 0:
            strike[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return strike
    
    def _filtered_noise(self, length, sample_rate, low_freq, high_freq):
        """Generate bandpass filtered white noise"""
        # Generate white noise
        noise = np.random.randn(length)
        
        # Simple butterworth-style filter approximation using FFT
        fft = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(length, 1/sample_rate)
        
        # Create bandpass filter
        filter_response = np.zeros(len(freqs))
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        filter_response[mask] = 1.0
        
        # Smooth the filter edges to prevent ringing
        transition_width = max((high_freq - low_freq) * 0.2, 50)  # At least 50 Hz transition
        for i, f in enumerate(freqs):
            if low_freq - transition_width < f < low_freq:
                # Smooth rise
                filter_response[i] = 0.5 * (1 + np.cos(np.pi * (low_freq - f) / transition_width))
            elif high_freq < f < high_freq + transition_width:
                # Smooth fall
                filter_response[i] = 0.5 * (1 + np.cos(np.pi * (f - high_freq) / transition_width))
        
        # Apply filter
        filtered_fft = fft * filter_response
        filtered_noise = np.fft.irfft(filtered_fft, length)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(filtered_noise))
        if max_val > 0:
            filtered_noise = filtered_noise / max_val * 0.95
        
        return filtered_noise
    
    def _generation_complete(self):
        """Called when audio generation is complete"""
        self.generate_btn.config(state=tk.NORMAL)
        if self.sounddevice_available:
            self.play_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)
        self.progress.stop()
        self.status_label.config(text="Sound generated successfully!", foreground="green")
    
    def _reset_ui(self):
        """Reset UI after error"""
        self.generate_btn.config(state=tk.NORMAL)
        self.progress.stop()
        self.status_label.config(text="Ready", foreground="blue")
    
    def play_sound(self):
        """Play the generated audio"""
        if not self.sounddevice_available:
            messagebox.showinfo("Info", "Install 'sounddevice' package for audio playback:\npip install sounddevice")
            return
        
        if self.audio_data is None:
            return
        
        self.is_playing = True
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Playing...", foreground="blue")
        
        def play_thread():
            try:
                self.sd.play(self.audio_data, self.sample_rate)
                self.sd.wait()
            except:
                pass
            finally:
                if self.is_playing:
                    self.root.after(0, self._playback_finished)
        
        self.playback_thread = threading.Thread(target=play_thread)
        self.playback_thread.start()
    
    def stop_sound(self):
        """Stop audio playback"""
        if self.sounddevice_available:
            self.sd.stop()
        self.is_playing = False
        self._playback_finished()
    
    def _playback_finished(self):
        """Called when playback finishes"""
        self.is_playing = False
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Playback finished", foreground="blue")
    
    def export_wav(self):
        """Export audio to WAV file"""
        if self.audio_data is None:
            messagebox.showwarning("Warning", "No audio to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export WAV File",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.status_label.config(text="Exporting...", foreground="blue")
                self.root.update()
                
                bit_depth = int(self.bit_depth_var.get())
                
                # Convert audio to appropriate bit depth
                if bit_depth == 16:
                    audio_int = np.int16(self.audio_data * 32767)
                    sample_width = 2
                elif bit_depth == 24:
                    audio_int = np.int32(self.audio_data * 8388607)
                    sample_width = 3
                else:  # 32-bit
                    audio_int = np.int32(self.audio_data * 2147483647)
                    sample_width = 4
                
                # Write WAV file
                with wave.open(filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(sample_width)
                    wav_file.setframerate(self.sample_rate)
                    
                    if bit_depth == 24:
                        # Special handling for 24-bit
                        audio_bytes = b''.join(
                            audio_int[i].to_bytes(3, byteorder='little', signed=True)
                            for i in range(len(audio_int))
                        )
                        wav_file.writeframes(audio_bytes)
                    else:
                        wav_file.writeframes(audio_int.tobytes())
                
                self.status_label.config(text=f"Exported successfully!", foreground="green")
                messagebox.showinfo("Success", f"Audio exported to:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
                self.status_label.config(text="Export failed", foreground="red")


def main():
    root = tk.Tk()
    app = CoastalSoundSynthesizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()