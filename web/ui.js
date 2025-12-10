export function createPreviewUI(nodePadding = 10) {
    // --- HTML Widget Setup ---
    const container = document.createElement("div");
    container.style.width = `calc(100% - ${nodePadding * 2}px)`;
    // container.style.height = "200px"; // Remove fixed height
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.backgroundColor = "#000";

    // --- STATIC PREVIEW MODE (Restored) ---
    // Back to flow layout so it pushes widgets instead of covering them
    container.style.position = "relative";
    container.style.marginTop = "10px";
    container.style.marginBottom = `${nodePadding}px`;
    container.style.marginLeft = `${nodePadding}px`;
    container.style.marginRight = `${nodePadding}px`;

    container.style.borderRadius = "8px";
    container.style.overflow = "hidden";
    container.style.border = "none";
    container.style.boxSizing = "border-box";

    // Height will be controlled by JavaScript in layout.js

    const canvas = document.createElement("canvas");
    canvas.style.top = "0";
    canvas.style.left = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%"; // Fill container
    canvas.style.display = "block";
    canvas.style.objectFit = "contain";
    container.appendChild(canvas);

    // --- Play/Pause Button ---
    const playBtn = document.createElement("div");
    playBtn.innerText = "⏸";
    playBtn.style.position = "absolute";
    playBtn.style.top = "5px";
    playBtn.style.right = "5px";
    playBtn.style.width = "24px";
    playBtn.style.height = "24px";
    playBtn.style.backgroundColor = "rgba(0,0,0,0.5)";
    playBtn.style.color = "white";
    playBtn.style.borderRadius = "50%";
    playBtn.style.textAlign = "center";
    playBtn.style.lineHeight = "24px";
    playBtn.style.cursor = "pointer";
    playBtn.style.fontSize = "12px";
    playBtn.style.zIndex = "10";
    playBtn.style.userSelect = "none";

    container.appendChild(playBtn);

    let isPlaying = false;
    playBtn.innerText = isPlaying ? "⏸" : "▶";

    playBtn.onclick = (e) => {
        e.stopPropagation(); // Prevent selecting node
        isPlaying = !isPlaying;
        playBtn.innerText = isPlaying ? "⏸" : "▶";
    };

    return {
        container,
        canvas,
        getIsPlaying: () => isPlaying
    };
}
