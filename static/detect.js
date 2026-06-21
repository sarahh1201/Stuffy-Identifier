const detectionList = document.getElementById("detection-list");

const nameEmoji = {
    Test: "🦊",
    Sniffy: "🐻",
    Dan: "🦕",
    Drake: "🐉",
    Cora: "🐮",
};

function renderDetections(detections) {
    if (!detections.length) {
        detectionList.innerHTML =
            '<div class="empty-state">Waiting for a stuffy to appear...</div>';
        return;
    }

    detectionList.innerHTML = detections
        .map((detection) => {
            const profileLink = detection.key
                ? `<a href="/animal/${detection.key}">View profile →</a>`
                : "";
            const confidence = Math.round(detection.confidence * 100);

            return `
                <div class="detection-chip">
                    <div>
                        <strong>${nameEmoji[detection.name] ? nameEmoji[detection.name] + " " : "🧸 "}${detection.name}</strong>
                        ${profileLink}
                    </div>
                    <span class="confidence">${confidence}%</span>
                </div>
            `;
        })
        .join("");
}

async function pollDetections() {
    try {
        const response = await fetch("/api/detections");
        const detections = await response.json();
        renderDetections(detections);
    } catch (error) {
        detectionList.innerHTML =
            '<div class="empty-state">Could not reach the detection service.</div>';
    }
}

pollDetections();
setInterval(pollDetections, 800);
