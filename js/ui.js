export function createPreviewUI(nodePadding = 3) {
    // --- HTML Widget Setup ---
    const container = document.createElement("div");
    container.style.width = `calc(100% - ${nodePadding * 2}px)`;
    container.style.height = "200px";
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.backgroundColor = "#202020";
    container.style.marginTop = "0px";
    container.style.marginBottom = `${nodePadding}px`;
    container.style.marginLeft = `${nodePadding}px`;
    container.style.marginRight = `${nodePadding}px`;
    container.style.borderRadius = "8px";
    container.style.overflow = "hidden";
    container.style.border = "none";
    container.style.boxSizing = "border-box";
    container.style.position = "relative"; // Make container relative so absolute button works

    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.display = "block";
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
