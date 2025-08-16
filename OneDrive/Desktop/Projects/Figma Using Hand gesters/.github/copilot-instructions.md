<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Figma Hand Gesture Control Project Instructions

## Project Overview

This project implements advanced browser automation for controlling Figma through hand gestures using computer vision and AI-enhanced recognition.

## Code Style Guidelines

- Use type hints for all function parameters and return values
- Follow PEP 8 conventions with Black formatter
- Use descriptive variable names that reflect the gesture or automation context
- Implement proper error handling for computer vision and browser automation failures

## Architecture Patterns

- Use dependency injection for browser automation engines (Selenium, Playwright, Puppeteer)
- Implement the Strategy pattern for different gesture recognition algorithms
- Use Observer pattern for real-time gesture event handling
- Apply Factory pattern for creating browser-specific automation instances

## Computer Vision Best Practices

- Always include confidence thresholds for gesture recognition
- Implement calibration methods for different lighting conditions
- Use frame buffering for smooth gesture tracking
- Include fallback mechanisms for failed gesture detection

## Browser Automation Guidelines

- Use explicit waits instead of implicit waits for better reliability
- Implement retry mechanisms for failed browser interactions
- Use page object pattern for Figma UI elements
- Include cross-browser compatibility checks

## Performance Considerations

- Optimize OpenCV and MediaPipe operations for real-time processing
- Use async/await for non-blocking browser operations
- Implement gesture prediction caching
- Monitor memory usage for long-running sessions

## Testing Requirements

- Include unit tests for gesture recognition accuracy
- Add integration tests for browser automation workflows
- Use mock objects for browser testing without actual browser instances
- Test gesture recognition under different lighting conditions

## Security Considerations

- Validate all user inputs for gesture calibration
- Sanitize any data sent to browser automation
- Implement safe mode for gesture recognition
- Include privacy controls for camera access
