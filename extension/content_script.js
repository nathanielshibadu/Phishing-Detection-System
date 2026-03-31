// Not required for popup flow, but kept for future automation.
// Example: send URL to background when page loads
chrome.runtime.sendMessage({ type: 'PAGE_LOADED', url: window.location.href });
