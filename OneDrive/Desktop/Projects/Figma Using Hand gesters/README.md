# Figma Hand Gesture Control

A minimal, optimized system for controlling Figma through hand gestures using computer vision and browser automation.

## Features

- **Real-time hand gesture recognition** using MediaPipe
- **Browser automation** with Selenium for Figma control
- **Optimized minimal codebase** for easy deployment
- **Cross-platform support** (Windows, macOS, Linux)

## Supported Gestures

| Gesture              | Action           | Description              |
| -------------------- | ---------------- | ------------------------ |
| Point (Index finger) | Select           | Select elements in Figma |
| Pinch                | Zoom             | Zoom in/out on canvas    |
| Open Palm            | Pan              | Pan around the canvas    |
| Fist                 | Delete           | Delete selected elements |
| Rectangle shape      | Create Rectangle | Draw rectangle           |
| Circle shape         | Create Circle    | Draw circle              |

## Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**

   ```bash
   python main.py
   ```

3. **Open Figma:**
   The application will automatically open Figma in your browser.

4. **Start gesturing:**
   Make gestures in front of your webcam to control Figma.

## Command Line Options

```bash
python main.py --browser chrome --headless  # Run with Chrome in headless mode
python main.py --browser firefox            # Use Firefox instead of Chrome
```

## Project Structure

```
figma_control/
├── main.py                 # Main application entry point
├── requirements.txt        # Minimal dependencies
├── figma_control/
│   ├── __init__.py        # Package initialization
│   ├── gesture_detector.py # Hand gesture detection
│   ├── browser_controller.py # Browser automation
│   └── gesture_mapper.py   # Gesture to action mapping
└── README.md              # This file
```

## System Requirements

- **Python 3.8+**
- **Webcam** (720p minimum)
- **Chrome or Firefox browser**
- **Good lighting** for gesture detection

## Troubleshooting

- **No gestures detected:** Check lighting and camera positioning
- **Browser automation fails:** Ensure Figma is loaded completely
- **Performance issues:** Close other applications using camera

## License

MIT License - feel free to modify and distribute.

1. **Clone the repository**

```bash
git clone <repository-url>
cd "Figma Using Hand gesters"
```

2. **Create virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install browser drivers**

```bash
playwright install
```

## Quick Start

1. **Basic gesture control**

```bash
python -m figma_control.main --mode basic
```

2. **Advanced AI mode**

```bash
python -m figma_control.main --mode advanced --ai-enhanced
```

3. **Multi-browser testing**

```bash
python -m figma_control.main --browser chrome,firefox --test-mode
```

## Gesture Commands

| Gesture     | Action           | Context     |
| ----------- | ---------------- | ----------- |
| Point       | Select Element   | General     |
| Pinch       | Create Rectangle | Design Mode |
| Circle      | Create Circle    | Design Mode |
| Swipe Left  | Undo             | General     |
| Swipe Right | Redo             | General     |
| Palm Open   | Pan Canvas       | Navigation  |
| Fist        | Delete Element   | Edit Mode   |

## Configuration

Edit `config/settings.yaml` to customize:

- Gesture sensitivity
- Browser preferences
- AI model settings
- Performance options

## Architecture

```
figma_control/
├── core/                 # Core gesture recognition
├── automation/           # Browser automation engines
├── ai/                  # AI-enhanced features
├── figma/               # Figma-specific controls
├── utils/               # Utilities and helpers
└── config/              # Configuration files
```

## Development

Run tests:

```bash
pytest tests/
```

Format code:

```bash
black figma_control/
```

Type checking:

```bash
mypy figma_control/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
