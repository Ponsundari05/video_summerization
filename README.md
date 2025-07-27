# 🎬 AI-Based Video Summarizer

This Python project automatically summarizes a video by extracting keyframes, scoring them, and stitching together the most important audio-visual segments into a shorter video. Perfect for reducing long videos into quick highlights.

---

## 📌 Features

- 🎥 Extract keyframes using frame difference analysis.
- 🧠 Score frames based on brightness and contrast.
- 🧩 Merge nearby scenes and remove overlaps.
- 🎞 Create summary video with audio using `moviepy`.
- ⚙️ Custom threshold, duration, and display options.

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install numpy opencv-python moviepy
# video_summerization
