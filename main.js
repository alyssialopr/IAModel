// Faire fonctionner le model onyx de reconnaissance de chiffres manuscrits avec ONNX runtime

const canvas = document.getElementById('drawing-canvas');
const clearButton = document.getElementById('clear-button');
const predictButton = document.getElementById('predict-button');
const resultElement = document.getElementById('result');

let isDrawing = false;
const ctx = canvas.getContext('2d');
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';
canvas.addEventListener('mousedown', () => { isDrawing = true; });
canvas.addEventListener('mouseup', () => { isDrawing = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', draw);
clearButton.addEventListener('click', clearCanvas);
predictButton.addEventListener('click', predictDigit);

function draw(event) {
    if (!isDrawing) return;
    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resultElement.textContent = 'Résultat : ';
}
