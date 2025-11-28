export const name = "Scaling";

export function apply(wrapper, box, isActive) {
    if (isActive) {
        wrapper.style.color = "#39E55F";
        wrapper.style.transform = "scale(1.2)";
        wrapper.style.margin = "0 5px";
    }
}