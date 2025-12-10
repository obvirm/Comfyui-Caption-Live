// color-widget.js - Color Picker via Right-Click Menu
// Simple and reliable - no canvas overlap issues

import { app } from "../../scripts/app.js";

const processedNodes = new Set();

app.registerExtension({
    name: "CaptionLive.ColorPicker",

    nodeCreated(node) {
        const title = node.title || "";
        const type = node.type || "";

        if (!title.includes("Caption") && !type.includes("Caption")) return;
        if (processedNodes.has(node.id)) return;
        processedNodes.add(node.id);

        // Add context menu options
        addContextMenuOptions(node);

        // Modify widget placeholder text to hint about right-click
        modifyColorWidgetHints(node);
    }
});

function modifyColorWidgetHints(node) {
    if (!node.widgets) return;

    const colorWidgetNames = ["highlight_color", "text_color", "stroke_color"];

    node.widgets.forEach(widget => {
        if (colorWidgetNames.includes(widget.name)) {
            // Add hint property (shown in some UIs)
            widget.options = widget.options || {};
            widget.options.tooltip = "Right-click node → Pick color";
        }
    });
}

function addContextMenuOptions(node) {
    const original = node.getExtraMenuOptions;

    node.getExtraMenuOptions = function (canvas, options) {
        if (original) original.call(this, canvas, options);

        const colorWidgets = this.widgets?.filter(w =>
            ["highlight_color", "text_color", "stroke_color"].includes(w.name)
        ) || [];

        if (colorWidgets.length > 0) {
            // Add separator
            options.push(null);

            // Add header-like item
            options.push({
                content: "━━━ 🎨 Color Picker ━━━",
                disabled: true
            });

            colorWidgets.forEach(w => {
                const color = w.value || "#FFFFFF";
                options.push({
                    content: `${getColorEmoji(color)} ${w.name.replace(/_/g, ' ')} (${color})`,
                    callback: () => openColorPicker(w, node)
                });
            });
        }
    };
}

function getColorEmoji(hex) {
    // Return a colored square unicode character based on the color
    // Since we can't do custom colors in menu, just use a generic icon
    return "🔵";
}

function openColorPicker(widget, node) {
    document.getElementById("clive-popup")?.remove();

    const current = widget.value || "#FFFFFF";

    const popup = document.createElement("div");
    popup.id = "clive-popup";
    popup.style.cssText = `
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0,0,0,0.7);
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: system-ui, -apple-system, sans-serif;
    `;

    const colorName = widget.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

    popup.innerHTML = `
        <div style="
            background: linear-gradient(145deg, #2d2d2d, #252525);
            border: 1px solid #444;
            border-radius: 20px;
            padding: 32px;
            min-width: 380px;
            box-shadow: 0 30px 100px rgba(0,0,0,0.8);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 28px;">
                <h3 style="margin: 0; color: #fff; font-size: 20px; font-weight: 600;">
                    🎨 ${colorName}
                </h3>
                <button id="clive-close" style="
                    background: #3a3a3a; border: none; color: #888; width: 36px; height: 36px;
                    border-radius: 10px; cursor: pointer; font-size: 24px; transition: all 0.2s;
                ">&times;</button>
            </div>
            
            <div style="display: flex; gap: 24px; margin-bottom: 28px;">
                <div style="position: relative;">
                    <input type="color" id="clive-color" value="${current}" style="
                        width: 120px; height: 120px; border: none; border-radius: 20px;
                        cursor: pointer; padding: 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                    ">
                </div>
                <div style="flex: 1; display: flex; flex-direction: column; justify-content: center;">
                    <div style="color: #666; font-size: 11px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px;">Hex Color Code</div>
                    <input type="text" id="clive-hex" value="${current}" style="
                        width: 100%; height: 60px; background: #1a1a1a; border: 2px solid #444;
                        border-radius: 14px; color: #fff; font-family: 'SF Mono', Consolas, monospace;
                        font-size: 26px; padding: 0 18px; text-transform: uppercase; box-sizing: border-box;
                        transition: border-color 0.2s;
                    ">
                    <div style="color: #555; font-size: 10px; margin-top: 8px;">Format: #RRGGBB</div>
                </div>
            </div>
            
            <div style="color: #666; font-size: 11px; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 1px;">Quick Select</div>
            <div id="clive-colors" style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 28px;"></div>
            
            <button id="clive-apply" style="
                width: 100%; height: 56px; 
                background: linear-gradient(135deg, #39E55F 0%, #2BC94D 100%);
                border: none; border-radius: 14px; color: #000; font-weight: 700; font-size: 17px;
                cursor: pointer; transition: transform 0.1s, box-shadow 0.2s;
                box-shadow: 0 4px 20px rgba(57, 229, 95, 0.3);
            ">✓ Apply Color</button>
        </div>
    `;

    document.body.appendChild(popup);

    const colorEl = document.getElementById("clive-color");
    const hexEl = document.getElementById("clive-hex");
    const colorsEl = document.getElementById("clive-colors");
    const applyEl = document.getElementById("clive-apply");
    const closeEl = document.getElementById("clive-close");

    // Preset colors
    const presets = [
        "#39E55F", "#FF6B6B", "#4ECDC4", "#FFE66D", "#F38181", "#AA96DA",
        "#FFFFFF", "#E8E8E8", "#888888", "#444444", "#222222", "#000000",
        "#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#007AFF", "#AF52DE"
    ];

    presets.forEach(c => {
        const s = document.createElement("div");
        s.style.cssText = `
            aspect-ratio: 1; background: ${c}; 
            border: 3px solid ${c.toUpperCase() === current.toUpperCase() ? '#fff' : 'transparent'};
            border-radius: 10px; cursor: pointer; 
            transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        `;
        s.onmouseenter = () => {
            s.style.transform = "scale(1.15)";
            s.style.boxShadow = "0 4px 16px rgba(0,0,0,0.4)";
        };
        s.onmouseleave = () => {
            s.style.transform = "scale(1)";
            s.style.boxShadow = "0 2px 8px rgba(0,0,0,0.2)";
        };
        s.onclick = () => {
            colorEl.value = c;
            hexEl.value = c;
            colorsEl.querySelectorAll("div").forEach(x => x.style.borderColor = "transparent");
            s.style.borderColor = "#fff";
        };
        colorsEl.appendChild(s);
    });

    // Events
    colorEl.oninput = () => {
        hexEl.value = colorEl.value.toUpperCase();
        updatePresets();
    };

    hexEl.oninput = () => {
        let v = hexEl.value.trim().toUpperCase();
        if (!v.startsWith("#")) v = "#" + v;
        hexEl.value = v;
        if (/^#[0-9A-Fa-f]{6}$/i.test(v)) {
            colorEl.value = v;
            hexEl.style.borderColor = "#444";
            updatePresets();
        } else {
            hexEl.style.borderColor = "#f55";
        }
    };

    hexEl.onfocus = () => hexEl.select();

    function updatePresets() {
        const cur = hexEl.value.toUpperCase();
        colorsEl.querySelectorAll("div").forEach(s => {
            const bg = s.style.background.toUpperCase();
            s.style.borderColor = bg === cur ? "#fff" : "transparent";
        });
    }

    applyEl.onclick = () => {
        const c = hexEl.value.toUpperCase();
        if (/^#[0-9A-Fa-f]{6}$/.test(c)) {
            widget.value = c;
            widget.callback?.(c);
            node.setDirtyCanvas?.(true, true);
            app.graph?.setDirtyCanvas?.(true, true);
        }
        popup.remove();
    };

    applyEl.onmouseenter = () => {
        applyEl.style.transform = "scale(1.02)";
        applyEl.style.boxShadow = "0 6px 30px rgba(57, 229, 95, 0.5)";
    };
    applyEl.onmouseleave = () => {
        applyEl.style.transform = "scale(1)";
        applyEl.style.boxShadow = "0 4px 20px rgba(57, 229, 95, 0.3)";
    };

    closeEl.onmouseenter = () => { closeEl.style.background = "#4a4a4a"; closeEl.style.color = "#fff"; };
    closeEl.onmouseleave = () => { closeEl.style.background = "#3a3a3a"; closeEl.style.color = "#888"; };
    closeEl.onclick = () => popup.remove();

    popup.onclick = (e) => { if (e.target === popup) popup.remove(); };

    document.addEventListener("keydown", function esc(e) {
        if (e.key === "Escape") { popup.remove(); document.removeEventListener("keydown", esc); }
        if (e.key === "Enter") { applyEl.click(); document.removeEventListener("keydown", esc); }
    });

    setTimeout(() => hexEl.select(), 100);
}

console.log("🎨 Color Picker: RIGHT-CLICK on node → Select color option");
console.log("   Tip: Look for '🎨 Color Picker' section in context menu");
