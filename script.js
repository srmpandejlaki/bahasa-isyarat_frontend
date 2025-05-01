const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvas');
const resultDisplay = document.getElementById('result');

let model = null;

// Mulai kamera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    console.error("Tidak bisa mengakses kamera:", err);
  }
}

// Load model TensorFlow.js (ganti path sesuai model kamu)
async function loadModel() {
  model = await tf.loadLayersModel('model/model.json');
  console.log("Model berhasil dimuat.");
}

// Proses deteksi
async function detectSign() {
  if (!model) return;

  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Preprocess gambar sesuai kebutuhan modelmu
  let imageTensor = tf.browser.fromPixels(canvas)
    .resizeNearestNeighbor([64, 64])
    .toFloat()
    .div(255.0)
    .expandDims();

  const prediction = model.predict(imageTensor);
  const predictionData = await prediction.data();
  const predictedIndex = predictionData.indexOf(Math.max(...predictionData));

  const labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
  resultDisplay.textContent = labels[predictedIndex];

  // (Opsional) Output suara
  const utter = new SpeechSynthesisUtterance(labels[predictedIndex]);
  speechSynthesis.speak(utter);

  imageTensor.dispose();
  prediction.dispose();
}

// Loop deteksi
setInterval(() => {
  detectSign();
}, 1000); // setiap 1 detik

// Inisialisasi
window.onload = async () => {
  await startCamera();
  await loadModel();
};
