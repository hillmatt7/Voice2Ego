
# Voice2Ego
> Analyze and segment audio recordings with precision.

**Voice2Ego** is a Python-based project designed for speaker diarization and audio analysis. It processes audio recordings to identify and segment different speakers, facilitating tasks such as transcription, speaker recognition, and audio analysis.

## Installing / Getting Started

### Prerequisites

Ensure you have the following installed on your system:

1. **Python** (v3.8 or higher)
2. **pip** (latest version)

### Steps to Set Up

To get Voice2Ego up and running locally:

```bash
# Clone the repository
git clone https://github.com/hillmatt7/Voice2Ego.git
cd Voice2Ego

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py --input /path/to/audio/file.wav --output /path/to/output/directory
```

Replace `/path/to/audio/file.wav` with the path to your audio file and `/path/to/output/directory` with the desired output directory.

## Features

### Core Functionality
- **Speaker Diarization**: Segments audio recordings to distinguish between different speakers.
- **Audio Analysis Utilities**: Provides tools for processing and analyzing audio data.

### Additional Features
- **Real-Time Segmentation**: View speaker segments dynamically as the script runs.
- **Error Reporting**: Detailed logs for troubleshooting and analysis.

## File Structure

### Key Files and Directories
- **`main.py`**: The main script for running the speaker diarization process.
- **`utils/`**: Contains utility functions and modules supporting the main script.
- **`test_speaker_split.py`**: Test cases for the speaker splitting functionality.
- **`requirements.txt`**: Lists the Python dependencies required for the project.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.

### Testing

To run the test suite:

```bash
pytest
```

This will execute the tests defined in `test_speaker_split.py` to ensure the functionality of the speaker splitting module.

## Developing

To contribute to or modify Voice2Ego, follow these steps:

```bash
# Clone the repository
git clone https://github.com/hillmatt7/Voice2Ego.git
cd Voice2Ego

# Install dependencies
pip install -r requirements.txt

# Start the development process
python main.py
```

### Running Tests

To ensure changes do not break functionality, run:

```bash
pytest
```

This command validates the integrity of the speaker splitting and diarization functionalities.

## Known Issues

- **Performance**: For large audio files, processing may take considerable time. Optimization efforts are ongoing.
- **Audio Formats**: Ensure your audio files are in a supported format (e.g., WAV) to avoid compatibility issues.

## Links

- **Project Homepage**: [Voice2Ego](https://github.com/hillmatt7/Voice2Ego)
- **Issue Tracker**: [Voice2Ego Issues](https://github.com/hillmatt7/Voice2Ego/issues)
- **Repository**: [Voice2Ego Repository](https://github.com/hillmatt7/Voice2Ego)

## Licensing

The code in this project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
