const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const statusElement = document.getElementById('status');
let model, faceCascade;

async function loadModel() {
    model = await tf.loadGraphModel('model/model.json');
    statusElement.innerText = 'Status: Model Loaded';
}

async function loadCascade() {
    try {
        const response = await fetch('/haar-cascade/haarcascade_frontalface_default.xml');
        if (!response.ok) {
            throw new Error(`Failed to load Haar Cascade XML file: ${response.statusText}`);
        }
        const xmlText = await response.text();
        const parser = new DOMParser();
        const xml = parser.parseFromString(xmlText, 'application/xml');
        if (xml.getElementsByTagName('parsererror').length > 0) {
            throw new Error('Error parsing Haar Cascade XML file');
        }
        
        faceCascade = new cv.CascadeClassifier();
        const cascadeFile = 'haar-cascade/haarcascade_frontalface_default.xml';

        const cascadeResponse = await fetch(cascadeFile);
        const buffer = await cascadeResponse.arrayBuffer();
        const data = new Uint8Array(buffer);
        cv.FS_createDataFile('/', 'haarcascade_frontalface_default.xml', data, true, false, false);
        faceCascade.load('haarcascade_frontalface_default.xml');
        console.log('Cascade loaded');

        statusElement.innerText = 'Status: Cascade Loaded';
    } catch (error) {
        console.error('Error loading cascade:', error);
        statusElement.innerText = 'Status: Error loading cascade';
    }
}

async function loadEmbeddings() {
    const response = await fetch('employee_embeddings/embeddings.json');
    return await response.json();
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

async function processFrame(embeddings) {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(image, gray, cv.COLOR_RGBA2GRAY);
    const faces = new cv.RectVector();
    const minSize = new cv.Size(50, 50);
    const maxSize = new cv.Size(0, 0);
    const flags = cv.CASCADE_FIND_BIGGEST_OBJECT;
    faceCascade.detectMultiScale(gray, faces, 1.1, 5, flags, minSize, maxSize);
    
    console.log('Faces: ' + faces.size());
    if (faces.size() === 0) {
        statusElement.innerText = 'No face detected';
    } else if (faces.size() > 1) {
        statusElement.innerText = 'More than one face detected';
    } else {

        const face = faces.get(0);
        const expandFactorV = 0.00;
        const expandFactorH = 0.0;
        const x = Math.max(0, face.x - face.width * expandFactorH);
        const y = Math.max(0, face.y - face.height * expandFactorV);
        const width = Math.min(image.cols - x, face.width * (1 + 2 * expandFactorH));
        const height = Math.min(image.rows - y, face.height * (1 + 2 * expandFactorV));

        const expandedFace = new cv.Rect(x, y, width, height);
        const faceImage = image.roi(expandedFace);
        // convert faceImage to RGB
        cv.cvtColor(faceImage, faceImage, cv.COLOR_RGBA2RGB);

        const imageWidth = faceImage.size().width; // 284
        const imageHeight = faceImage.size().height; // 284
        const imageChannels = faceImage.channels(); // 3

        // console.log("Image Width: " + imageWidth);
        // console.log("Image Height: " + imageHeight);
        // console.log("Image Channels: " + imageChannels);
        
        const faceImageArray = faceImage.data; 
        // console.log("Face Image Array:")
        // console.log(faceImageArray); 

        const reshapedArray = new Uint8Array(imageWidth * imageHeight * imageChannels);
        for (let i = 0; i < faceImageArray.length; i++) {
            reshapedArray[i] = faceImageArray[i];
        }

        const tensor = tf.tensor3d(reshapedArray, [imageHeight, imageWidth, imageChannels])
                                    .resizeBilinear([224, 224])
                                    .expandDims(0)
                                    .toFloat()
                                    .div(255)
                                    .transpose([0, 3, 1, 2]);

        
        // const tensorArray = await tensor.array();
        // console.log("Tensor Array:")
        // console.log(tensorArray);

        const mean = tf.tensor([0.485, 0.456, 0.406]).expandDims(0).expandDims(2).expandDims(3);    
        const std = tf.tensor([0.229, 0.224, 0.225]).expandDims(0).expandDims(2).expandDims(3);

        const normalised = tensor.sub(mean).div(std);
        // const normalisedArray = await normalised.array();
        // console.log("Normalised Array:")
        // console.log(normalisedArray);

        const embedding = model.execute(normalised).dataSync();
        let maxSimilarity = 0;
        let maxSimilarityEmployee = '';
        let verified = false;

        statusElement.innerText = 'Checking authorization...';

        for (const employee of embeddings) {
            for (const empEmbedding of employee.embeddings) {
                const similarity = cosineSimilarity(embedding, empEmbedding);
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    maxSimilarityEmployee = employee.name;
                }
                if (similarity > 0.55) {
                    if (similarity === maxSimilarity) {
                        statusElement.innerText = `Identity Verified: ${employee.name}`;
                    }
                    verified = true;
                }
            }
        }

        if (!verified) {
            statusElement.innerText = 'Not authorized';
            console.log(`Max Similarity: ${maxSimilarity*100}%`);
        }
        if (verified) {
            console.log(`Person with max similarity: ${maxSimilarityEmployee}`);    
            console.log(`Max similarity: ${maxSimilarity*100}%`);
            }
        }  
}

async function init() {
    await tf.setBackend('cpu');
    console.log(`Backend set to ${tf.getBackend()}`);

    await loadModel();
    await loadCascade();
    const embeddings = await loadEmbeddings();
    await setupCamera();

    statusElement.innerText = 'Status: Ready';

    video.addEventListener('play', () => {
        const frameRate = 8;
        let frameCount = 0;

        setInterval(() => {
            if (video.paused || video.ended) return;

            frameCount++;
            if (frameCount % frameRate !== 0) return;

            processFrame(embeddings);
        }, 100);
    });
}

function cosineSimilarity(a, b) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;

    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

document.addEventListener('DOMContentLoaded', init);