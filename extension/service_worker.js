// Minimal service worker
self.addEventListener('install', (event) => {
  console.log('PhishGuard service worker installed');
});

self.addEventListener('activate', (event) => {
  console.log('PhishGuard service worker activated');
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('SW received message', message);
});
