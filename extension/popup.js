// Open Admin Dashboard
document.getElementById("openDashboard").addEventListener("click", () => {
    chrome.tabs.create({
        url: chrome.runtime.getURL("admin/dashboard.html")
    });
});

// Scan current active tab
document.getElementById("scanBtn").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const url = tabs[0].url;
        analyzeURL(url);
    });
});

// Call local Flask API
function analyzeURL(url) {
    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ url: url })
    })
    .then(response => response.json())
    .then(data => {
        displayResult(data);

        // Log only phishing results
        if (data.label === "phishing") {
            chrome.storage.local.get(["phishingLogs"], (res) => {
                const logs = res.phishingLogs || [];
                logs.push({
                    url: data.url,
                    confidence: data.confidence,
                    timestamp: new Date().toISOString()
                });
                chrome.storage.local.set({ phishingLogs: logs });
            });
        }
    })
    .catch(() => {
        const result = document.getElementById("result");
        result.className = "unknown";
        result.innerText = "Error: Could not reach PhishGuard API";
    });
}

// Update popup UI with API result
function displayResult(data) {
    const resultDiv = document.getElementById("result");

    if (data.label === "phishing") {
        resultDiv.className = "phishing";
        resultDiv.innerHTML = `
            ⚠️ <strong>Phishing Detected!</strong><br>
            Confidence: ${(data.confidence * 100).toFixed(1)}%
        `;
    } 
    else if (data.label === "legit") {
        resultDiv.className = "legit";
        resultDiv.innerHTML = `
            ✅ <strong>Website is Safe</strong><br>
            Confidence: ${(data.confidence * 100).toFixed(1)}%
        `;
    } 
    else {
        resultDiv.className = "unknown";
        resultDiv.innerText = "Unknown response from server";
    }
}
