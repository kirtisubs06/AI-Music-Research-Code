# Music Emotion Analyzer

## Introduction
Music Emotion Analyzer is a Python application designed to analyze audio files, particularly music tracks, to extract various audio features and categorize them based on the emotional content. It uses `librosa` for audio processing and machine learning techniques for emotion categorization.

## Features

**Audio Processing:**
- Load and process audio files using the `librosa` library.
- Extract key audio features like tempo, harmonic components, and volume levels.

**Emotional Analysis:**
- Analyze the emotional content of music tracks based on extracted audio features.
- Utilize machine learning techniques to categorize tracks into emotions like calm, joyous, melancholy, and restlessness.

**Key Detection:**
- Implement the Krumhansl-Schmuckler key-finding algorithm to determine the musical key of a track.
- Generate chromagrams to visually represent the intensity of pitch classes over time.

**Data Visualization:**
- Visualize audio features and emotional categorizations using `matplotlib`.
- Perform dimensionality reduction using PCA for effective visualization.

**Customizable Analysis:**
- Analyze specific segments of tracks for detailed insights.
- Extendable to include additional audio features or emotional categories.



