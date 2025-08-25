"""
Browser automation controller for Figma.
Supports Selenium and Playwright with optimized Figma-specific actions.
"""

import asyncio
from typing import Optional, Dict, Any, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

# Simple gesture result class for compatibility


class GestureResult:
    """Simplified gesture result for browser actions."""

    def __init__(self, gesture: str, confidence: float = 0.0, position: Tuple[int, int] = (0, 0)):
        self.gesture = gesture
        self.confidence = confidence
        self.position = position


class BrowserController:
    """Optimized browser controller for Figma automation."""

    def __init__(self, browser_type: str = "chrome", headless: bool = False):
        self.browser_type = browser_type
        self.headless = headless
        self.driver: Optional[webdriver.Remote] = None
        self.wait: Optional[WebDriverWait] = None
        self.actions: Optional[ActionChains] = None

        # Figma-specific selectors (these may need updates based on Figma's DOM)
        self.selectors = {
            "canvas": "[data-testid='canvas']",
            "toolbar": "[data-testid='toolbar']",
            "properties_panel": "[data-testid='right-sidebar']",
            "layers_panel": "[data-testid='left-sidebar']"
        }

    async def initialize(self):
        """Initialize browser driver."""
        try:
            if self.browser_type.lower() in ["chrome", "brave"]:
                options = ChromeOptions()
                if self.headless:
                    options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-web-security")
                options.add_argument("--allow-running-insecure-content")

                # For Brave browser, try to find Brave executable
                if self.browser_type.lower() == "brave":
                    import os
                    brave_paths = [
                        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
                        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
                        r"C:\Users\{}\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe".format(
                            os.getenv('USERNAME'))
                    ]
                    for path in brave_paths:
                        if os.path.exists(path):
                            options.binary_location = path
                            break

                self.driver = webdriver.Chrome(
                    service=webdriver.chrome.service.Service(
                        ChromeDriverManager().install()),
                    options=options
                )

            elif self.browser_type == "firefox":
                options = FirefoxOptions()
                if self.headless:
                    options.add_argument("--headless")

                self.driver = webdriver.Firefox(
                    service=webdriver.firefox.service.Service(
                        GeckoDriverManager().install()),
                    options=options
                )

            self.wait = WebDriverWait(self.driver, 10)
            self.actions = ActionChains(self.driver)
            self.driver.maximize_window()

        except Exception as e:
            raise Exception(f"Failed to initialize browser: {e}")

    async def navigate_to_figma(self, file_url: Optional[str] = None):
        """Navigate to Figma."""
        if file_url:
            self.driver.get(file_url)
        else:
            self.driver.get("https://www.figma.com")

        # Wait for page to load
        await asyncio.sleep(2)

    async def execute_action(self, action: str, gesture: GestureResult):
        """Execute Figma action based on gesture."""
        try:
            canvas = self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.selectors["canvas"]))
            )

            x, y = gesture.position

            if action == "select":
                await self._click_at_position(canvas, x, y)

            elif action == "create_rectangle":
                await self._create_rectangle(canvas, x, y)

            elif action == "create_circle":
                await self._create_circle(canvas, x, y)

            elif action == "move":
                await self._drag_element(canvas, x, y)

            elif action == "delete":
                await self._delete_selected()

            elif action == "undo":
                await self._undo()

            elif action == "zoom":
                await self._zoom(canvas, x, y)

            elif action == "pan":
                await self._pan_canvas(canvas, x, y)

        except Exception as e:
            print(f"Action execution failed: {e}")

    async def _click_at_position(self, canvas, x: int, y: int):
        """Click at specific position on canvas."""
        self.actions.move_to_element_with_offset(
            canvas, x, y).click().perform()

    async def _create_rectangle(self, canvas, x: int, y: int):
        """Create rectangle tool and draw."""
        # Select rectangle tool (R key)
        self.driver.find_element(By.TAG_NAME, "body").send_keys("r")
        await asyncio.sleep(0.1)

        # Draw rectangle
        self.actions.move_to_element_with_offset(canvas, x, y)\
            .click_and_hold()\
            .move_by_offset(100, 100)\
            .release()\
            .perform()

    async def _create_circle(self, canvas, x: int, y: int):
        """Create circle tool and draw."""
        # Select ellipse tool (O key)
        self.driver.find_element(By.TAG_NAME, "body").send_keys("o")
        await asyncio.sleep(0.1)

        # Draw circle
        self.actions.move_to_element_with_offset(canvas, x, y)\
            .click_and_hold()\
            .move_by_offset(80, 80)\
            .release()\
            .perform()

    async def _drag_element(self, canvas, x: int, y: int):
        """Drag selected element."""
        # Assume element is already selected
        self.actions.move_to_element_with_offset(canvas, x, y)\
            .click_and_hold()\
            .move_by_offset(50, 50)\
            .release()\
            .perform()

    async def _delete_selected(self):
        """Delete selected element."""
        self.driver.find_element(By.TAG_NAME, "body").send_keys(
            "\ue017")  # Delete key

    async def _undo(self):
        """Undo last action."""
        # Ctrl+Z for undo
        self.actions.key_down('\ue009').send_keys(
            'z').key_up('\ue009').perform()

    async def _zoom(self, canvas, x: int, y: int):
        """Zoom in/out at position."""
        # Scroll wheel simulation for zoom
        self.actions.move_to_element_with_offset(canvas, x, y)\
            .scroll_by_amount(0, 3)\
            .perform()

    async def _pan_canvas(self, canvas, x: int, y: int):
        """Pan the canvas."""
        # Space + drag for panning
        self.actions.key_down(' ')\
            .move_to_element_with_offset(canvas, x, y)\
            .click_and_hold()\
            .move_by_offset(30, 30)\
            .release()\
            .key_up(' ')\
            .perform()

    async def close(self):
        """Close browser."""
        if self.driver:
            self.driver.quit()
