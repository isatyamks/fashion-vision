// Minimal cropper for the captured frame
window.onload = function() {
    const urlParams = new URLSearchParams(window.location.search);
    const imgData = urlParams.get('img');
    const img = document.getElementById('frame-img');
    img.src = imgData;

    // Simple crop: select area with mouse
    let startX, startY, endX, endY, cropping = false;
    let cropRect = document.createElement('div');
    cropRect.style.position = 'absolute';
    cropRect.style.border = '2px dashed #3897f0';
    cropRect.style.pointerEvents = 'none';
    cropRect.style.display = 'none';
    document.body.appendChild(cropRect);

    img.onmousedown = function(e) {
        cropping = true;
        startX = e.pageX;
        startY = e.pageY;
        cropRect.style.left = startX + 'px';
        cropRect.style.top = startY + 'px';
        cropRect.style.width = '0px';
        cropRect.style.height = '0px';
        cropRect.style.display = 'block';
    };
    img.onmousemove = function(e) {
        if (!cropping) return;
        endX = e.pageX;
        endY = e.pageY;
        cropRect.style.width = Math.abs(endX - startX) + 'px';
        cropRect.style.height = Math.abs(endY - startY) + 'px';
        cropRect.style.left = Math.min(startX, endX) + 'px';
        cropRect.style.top = Math.min(startY, endY) + 'px';
    };
    img.onmouseup = function(e) {
        cropping = false;
    };
    document.getElementById('download-btn').onclick = function() {
        // For demo: just download the full image (cropping logic can be improved)
        const a = document.createElement('a');
        a.href = img.src;
        a.download = 'cropped_frame.png';
        a.click();
    };
};
