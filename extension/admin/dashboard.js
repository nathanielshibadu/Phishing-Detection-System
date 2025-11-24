function loadLogs() {
    chrome.storage.local.get(["phishingLogs"], (result) => {
        const logs = result.phishingLogs || [];
        const tbody = document.querySelector("#logTable tbody");
        tbody.innerHTML = "";

        logs.forEach((log, index) => {
            const tr = document.createElement("tr");

            tr.innerHTML = `
                <td>${log.url}</td>
                <td class="danger">${(log.confidence * 100).toFixed(1)}%</td>
                <td>${new Date(log.timestamp).toLocaleString()}</td>
                <td><button data-index="${index}" class="deleteBtn">Delete</button></td>
            `;

            tbody.appendChild(tr);
        });
    });
}

document.getElementById("clearLogs").addEventListener("click", () => {
    chrome.storage.local.set({ phishingLogs: [] }, loadLogs);
});

document.addEventListener("click", (e) => {
    if (e.target.classList.contains("deleteBtn")) {
        const index = e.target.getAttribute("data-index");

        chrome.storage.local.get(["phishingLogs"], (result) => {
            const logs = result.phishingLogs || [];
            logs.splice(index, 1);
            chrome.storage.local.set({ phishingLogs: logs }, loadLogs);
        });
    }
});

loadLogs();
