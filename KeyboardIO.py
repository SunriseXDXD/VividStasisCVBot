import pyautogui
from typing import Dict, Set

class Key:
    def __init__(self, value: str):
        self.value = value
        self.is_pressed = False
        self.was_pressed = False
    
    def press(self):
        if not self.is_pressed:
            pyautogui.keyDown(self.value)
            self.is_pressed = True
    
    def release(self):
        if self.is_pressed:
            pyautogui.keyUp(self.value)
            self.is_pressed = False
    
    def update_state(self, should_press: bool):
        self.was_pressed = self.is_pressed
        if should_press:
            self.press()
        else:
            self.release()

class KeyboardIO:
    def __init__(self):
        # Initialize keys for 4K rhythm game (left to right: D, F, J, K)
        self.keys = {
            'd': Key('d'),  # Leftmost key
            'f': Key('f'),  # Left center key
            'j': Key('j'),  # Right center key
            'k': Key('k')   # Rightmost key
        }
        
        # Group keys for convenience
        self.left_keys = {'d', 'f'}
        self.right_keys = {'j', 'k'}
        
        # Track pressed keys to avoid redundant input
        self.currently_pressed: Set[str] = set()
    
    def update(self, key_states: Dict[str, bool]):
        """
        Update the state of all keys based on the input dictionary.
        
        Args:
            key_states: Dictionary mapping key names (d/f/j/k) to their desired state (True = pressed)
        """
        if not isinstance(key_states, dict):
            raise ValueError("key_states must be a dictionary")
            
        # Update each key's state
        for key_name, key_obj in self.keys.items():
            should_press = key_states.get(key_name, False)
            key_obj.update_state(should_press)
        
        # Update currently pressed keys set
        self.currently_pressed = {k for k, v in key_states.items() if v}
    
    def press_key(self, key_name: str):
        """Press a single key if it exists and isn't already pressed"""
        if key_name in self.keys and not self.keys[key_name].is_pressed:
            self.keys[key_name].press()
    
    def release_key(self, key_name: str):
        """Release a single key if it exists and is currently pressed"""
        if key_name in self.keys and self.keys[key_name].is_pressed:
            self.keys[key_name].release()
    
    def release_all(self):
        """Release all keys"""
        for key in self.keys.values():
            key.release()
        self.currently_pressed.clear()
    
    def __del__(self):
        """Ensure all keys are released when the object is destroyed"""
        self.release_all()