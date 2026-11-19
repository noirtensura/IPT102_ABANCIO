"use strict";

const input = document.getElementById("skin-photo");
const choosePhoto = document.getElementById("choose-photo");
const preview = document.getElementById("photo-preview");
const example = document.getElementById("example-label");
const empty = document.getElementById("empty-photo");
const removePhoto = document.getElementById("remove-photo");
const checkImage = document.getElementById("check-image");
const scannerMessage = document.getElementById("scanner-message");
let previewUrl = "";

function setMessage(message) {
    scannerMessage.textContent = message;
    scannerMessage.hidden = !message;
}

function clearPhoto() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = "";
    input.value = "";
    preview.hidden = true;
    preview.removeAttribute("src");
    example.hidden = true;
    removePhoto.hidden = true;
    empty.hidden = false;
    checkImage.disabled = true;
}

choosePhoto.addEventListener("click", () => input.click());
removePhoto.addEventListener("click", () => {
    clearPhoto();
    setMessage("");
    choosePhoto.focus();
});

input.addEventListener("change", () => {
    const photo = input.files[0];
    if (!photo) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(photo);
    checkImage.disabled = true;
    preview.alt = "Selected skin photo";
    preview.classList.remove("example-photo");
    preview.src = previewUrl;
    preview.hidden = false;
    example.hidden = true;
    empty.hidden = true;
    removePhoto.hidden = false;
    setMessage("");
});

preview.addEventListener("load", () => {
    if (previewUrl && preview.currentSrc === previewUrl) checkImage.disabled = false;
});
preview.addEventListener("error", () => {
    if (!previewUrl) return;
    clearPhoto();
    setMessage("This photo could not be opened. Choose a readable image file.");
});

const dialog = document.getElementById("camera-dialog");
const form = document.querySelector("form.workspace");
const checkLabelDefault = checkImage ? checkImage.innerHTML : "";
if (form && checkImage) {
    form.addEventListener("submit", () => {
        if (!input.files || !input.files.length) return;
        checkImage.disabled = true;
        checkImage.setAttribute("aria-busy", "true");
        checkImage.textContent = "Analyzing image...";
    });
    window.addEventListener("pageshow", () => {
        if (checkLabelDefault) checkImage.innerHTML = checkLabelDefault;
        checkImage.removeAttribute("aria-busy");
        checkImage.disabled = !(input.files && input.files.length);
    });
}
const video = document.getElementById("camera-video");
const cameraMessage = document.getElementById("camera-message");
const capture = document.getElementById("capture-photo");
let cameraAttempt = 0;

function stopCamera() {
    cameraAttempt++;
    video.srcObject?.getTracks().forEach(track => track.stop());
    video.srcObject = null;
}

document.getElementById("open-camera").addEventListener("click", async () => {
    const attempt = ++cameraAttempt;
    video.hidden = true;
    capture.disabled = true;
    cameraMessage.textContent = "Allow camera access to take a photo.";
    dialog.showModal();
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } } });
        if (!dialog.open || attempt !== cameraAttempt) {
            stream.getTracks().forEach(track => track.stop());
            return;
        }
        video.srcObject = stream;
        video.hidden = false;
        await video.play();
        if (!dialog.open || attempt !== cameraAttempt) return;
        capture.disabled = false;
        cameraMessage.textContent = "";
    } catch {
        if (!dialog.open || attempt !== cameraAttempt) return;
        stopCamera();
        video.hidden = true;
        cameraMessage.textContent = "Camera unavailable. Close this window to upload a photo instead.";
    }
});

capture.addEventListener("click", () => {
    if (!video.videoWidth || !video.videoHeight) return;
    const attempt = cameraAttempt;
    capture.disabled = true;
    const frame = document.createElement("canvas");
    frame.width = video.videoWidth;
    frame.height = video.videoHeight;
    frame.getContext("2d").drawImage(video, 0, 0);
    frame.toBlob(photo => {
        if (!dialog.open || attempt !== cameraAttempt) return;
        if (!photo) {
            capture.disabled = false;
            cameraMessage.textContent = "The photo could not be captured. Try again.";
            return;
        }
        const selection = new DataTransfer();
        selection.items.add(new File([photo], "camera-photo.jpg", { type: "image/jpeg" }));
        input.files = selection.files;
        input.dispatchEvent(new Event("change"));
        dialog.close();
    }, "image/jpeg", 0.92);
});

dialog.addEventListener("close", stopCamera);
window.addEventListener("pagehide", stopCamera);
