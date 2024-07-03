from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Data informasi tentang diabetes
diabetes_info = {
    'attributes': [
        {'name': 'Pregnancies', 'description': 'Berapa kali hamil'},
        {'name': 'Glucose', 'description': 'Konsentrasi glukosa plasma selama 2 jam'},
        {'name': 'BloodPressure', 'description': 'Tekanan darah diastolik (mm Hg)'},
        {'name': 'SkinThickness', 'description': 'Ketebalan kulit trisep (mm)'},
        {'name': 'Insulin', 'description': 'Insulin serum selama 2 jam (mu U/ml)'},
        {'name': 'BMI', 'description': 'Index massa tubuh (berat dalam kg/(tinggi dalam m)^2)'},
        {'name': 'DiabetesPedigreeFunction', 'description': 'Fungsi silsilah diabetes'},
        {'name': 'Age', 'description': 'Usia (tahun)'},
        {'name': 'Outcome', 'description': 'Hasil (0: Tidak Menderita Diabetes, 1: Menderita Diabetes)'}
    ],
    'about': {
        'definition': 'Diabetes adalah kondisi kronis yang terjadi ketika tubuh tidak dapat memproduksi atau menggunakan insulin secara efektif.',
        'types': [
            {'name': 'Diabetes Tipe 1', 'description': 'Penyakit autoimun di mana sistem kekebalan tubuh menyerang sel beta pankreas.'},
            {'name': 'Diabetes Tipe 2', 'description': 'Tubuh tidak dapat menggunakan insulin dengan baik dan sering dikaitkan dengan obesitas.'},
            {'name': 'Diabetes Gestasional', 'description': 'Diabetes yang terjadi selama kehamilan dan biasanya hilang setelah melahirkan.'}
        ],
        'symptoms': [
            'Sering merasa haus.',
            'Sering buang air kecil.',
            'Penurunan berat badan tanpa sebab.',
            'Kelelahan ekstrem.',
            'Penglihatan kabur.',
            'Luka yang sembuhnya lama.',
            'Infeksi yang sering terjadi.'
        ],
        'prevention': [
            'Makan makanan yang sehat dan seimbang.',
            'Aktivitas fisik rutin setidaknya 150 menit per minggu.',
            'Menjaga berat badan yang sehat.',
            'Menghindari konsumsi gula berlebihan.',
            'Memeriksa kadar gula darah secara teratur.'
        ],
        'risk_factors': [
            'Obesitas atau kelebihan berat badan.',
            'Kurangnya aktivitas fisik.',
            'Riwayat keluarga dengan diabetes.',
            'Usia di atas 45 tahun.',
            'Tekanan darah tinggi atau kolesterol tinggi.',
            'Riwayat diabetes gestasional.',
            'Diet tinggi gula dan lemak jenuh.',
            'Merokok dan konsumsi alkohol berlebihan.'
        ],
        'complications': [
            'Penyakit jantung dan stroke.',
            'Kerusakan saraf (neuropati), terutama di kaki.',
            'Masalah ginjal (nefropati).',
            'Masalah mata seperti retinopati diabetik.',
            'Masalah kaki seperti ulkus dan infeksi.',
            'Masalah kulit seperti infeksi bakteri dan jamur.',
            'Depresi dan masalah kesehatan mental.'
        ],
        'treatment': [
            'Pengendalian kadar gula darah melalui diet dan olahraga.',
            'Obat oral seperti metformin.',
            'Terapi insulin untuk beberapa pasien.',
            'Pemantauan kadar gula darah secara teratur.',
            'Pendidikan diabetes dan dukungan psikososial.'
        ],
        'statistics': {
            'global': 'Lebih dari 422 juta orang di dunia hidup dengan diabetes. Penyakit ini adalah penyebab utama kematian di banyak negara.',
            'regional': {
                'asia': 'Asia memiliki salah satu tingkat diabetes tertinggi di dunia, terutama di negara-negara seperti India dan Cina.',
                'africa': 'Di Afrika, prevalensi diabetes terus meningkat seiring urbanisasi dan perubahan gaya hidup.',
                'europe': 'Di Eropa, sekitar 60 juta orang hidup dengan diabetes, dan prevalensinya terus meningkat.',
                'americas': 'Di Amerika Serikat, sekitar 34 juta orang, atau 10,5% dari populasi, hidup dengan diabetes.'
            }
        },
        'resources': [
            {'name': 'WHO Diabetes', 'url': 'https://www.who.int/news-room/fact-sheets/detail/diabetes'},
            {'name': 'American Diabetes Association', 'url': 'https://www.diabetes.org/'},
            {'name': 'International Diabetes Federation', 'url': 'https://www.idf.org/'},
            {'name': 'Diabetes UK', 'url': 'https://www.diabetes.org.uk/'}
        ]
    }
}

# Load model
model = joblib.load('DTRModel.pkl')


@app.route('/')
def index():
    return render_template('index.html', diabetes_info=diabetes_info)

@app.route('/about')
def about():
    return render_template('about.html', diabetes_info=diabetes_info)

@app.route('/faq')
def baik_buruk():
    return render_template('faq.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        input_data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetes_pedigree_function']),
            float(request.form['age'])
        ]

        # Convert input to numpy array
        input_array = np.array([input_data])

        # Perform prediction
        prediction = model.predict(input_array)

        # Determine prediction result
        result = 'Menderita Diabetes' if prediction[0] >= 0.5 else 'Tidak Menderita Diabetes'

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/faq_answer', methods=['POST'])
def faq_answer():
    try:
        question = request.form['question']

        # Mapping questions to answers
        faq_responses = {
            "Apa itu diabetes mellitus?": "Diabetes mellitus adalah kondisi kronis di mana tubuh tidak dapat mengatur kadar gula darah secara efektif.",
            "Apa perbedaan diabetes tipe 1 dan tipe 2?": "Diabetes Tipe 1 terjadi ketika tubuh tidak memproduksi insulin, sementara diabetes Tipe 2 terjadi ketika tubuh tidak dapat menggunakan insulin dengan baik.",
            "Apa penyebab diabetes mellitus?": "Diabetes mellitus dapat disebabkan oleh faktor genetik, gaya hidup yang tidak sehat, dan faktor risiko lainnya seperti obesitas dan kurangnya aktivitas fisik.",
            "Apa gejala umum diabetes mellitus?": "Gejala umum diabetes mellitus termasuk sering merasa haus, sering buang air kecil, penurunan berat badan tanpa sebab, dan kelelahan ekstrem.",
            "Bagaimana cara mencegah diabetes mellitus?": "Cara terbaik untuk mencegah diabetes mellitus adalah dengan menjaga pola makan sehat, rutin berolahraga, dan menjaga berat badan yang sehat.",
            "Apakah diabetes mellitus dapat disembuhkan?": "Diabetes mellitus tidak dapat disembuhkan, tetapi dapat dikelola dengan perawatan yang tepat seperti diet, olahraga, dan obat-obatan.",
            "Bagaimana cara mendiagnosis diabetes mellitus?": "Diabetes mellitus biasanya didiagnosis melalui tes darah seperti tes gula darah puasa dan tes HbA1c.",
            "Apa komplikasi yang bisa terjadi akibat diabetes mellitus?": "Komplikasi diabetes mellitus termasuk penyakit jantung, kerusakan saraf, kerusakan ginjal, masalah penglihatan, dan luka yang sulit sembuh."
        }

        # Find the answer or return a default message
        response = faq_responses.get(question, "Pertanyaan tidak ditemukan atau belum ada di sistem kami. Silakan coba pertanyaan lain.")

        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/baru')
def baru():
    return render_template('baru.html')

if __name__ == '__main__':
    app.run(debug=True)
