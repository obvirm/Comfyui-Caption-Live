export const name = "Box (TikTok)";

export function apply(wrapper, box, isActive) {
    if (isActive) {
        wrapper.style.color = "white";
        box.style.opacity = "1";
        box.style.transform = "scale(1)";
    }
}