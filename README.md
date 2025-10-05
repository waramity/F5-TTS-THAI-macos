# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

## Support For Thai Language (macOS Version)

Text-to-Speech (TTS) ภาษาไทย — เครื่องมือสร้างเสียงพูดจากข้อความด้วยเทคนิค Flow Matching ด้วยโมเดล F5-TTS

**โมเดล Finetune**: VIZINTZOR/F5-TTS-THAI  
**โมเดล Finetune V2 (IPA)**: VIZINTZOR/F5-TTS-TH-V2

⚠️ **หมายเหตุ**: การอ่านข้อความยาวๆ หรือบางคำ ยังไม่ถูกต้อง

---

## ความต้องการของระบบ

- **macOS**: 12.3 ขึ้นไป (สำหรับ MPS support)
- **Python**: 3.10 ขึ้นไป
- **Apple Silicon (M1/M2/M3)**: แนะนำสำหรับประสิทธิภาพที่ดีที่สุด
- **eSpeak NG**: สำหรับการประมวลผลเสียง

---

## การติดตั้ง

### 1. ติดตั้ง Homebrew (ถ้ายังไม่มี)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. ติดตั้ง Python และ eSpeak NG
```bash
brew install python@3.10 espeak-ng
```

### 3. ติดตั้ง F5-TTS-THAI
```bash
# Clone repository
git clone https://github.com/VYNCX/F5-TTS-THAI.git
cd F5-TTS-THAI

# สร้าง virtual environment
python3 -m venv venv

# เปิดใช้งาน virtual environment
source venv/bin/activate

# อัพเกรด pip
pip install --upgrade pip

# ติดตั้ง F5-TTS-THAI
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git

# ติดตั้ง PyTorch สำหรับ Apple Silicon (M1/M2/M3)
pip install torch torchaudio
```

### 4. ตรวจสอบการติดตั้ง
```bash
# ตรวจสอบว่า MPS (Metal Performance Shaders) พร้อมใช้งาน
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

## การใช้งาน

### เปิดใช้งาน Web UI

```bash
# ตรวจสอบว่าอยู่ใน virtual environment (ควรเห็น (venv) ที่หน้า terminal)
# ถ้ายังไม่ได้เปิดใช้งาน ให้รัน: source venv/bin/activate

# รัน Web UI
f5-tts_webui
```

หรือ

```bash
python src/f5_tts/f5_tts_webui.py
```

เปิดเบราว์เซอร์และไปที่: **http://127.0.0.1:7860**

### การใช้งานประจำวัน

ทุกครั้งที่ต้องการใช้งาน:

```bash
# 1. ไปยังโฟลเดอร์โปรเจค
cd ~/path/to/F5-TTS-THAI

# 2. เปิดใช้งาน virtual environment
source venv/bin/activate

# 3. รัน Web UI
f5-tts_webui

# 4. เมื่อใช้งานเสร็จ ปิด virtual environment
deactivate
```

---

## ใช้งานบน Google Colab

[Google Colab Notebook](link-to-your-colab)

---

## คำแนะนำการใช้งานบน macOS

### สำหรับ Apple Silicon (M1/M2/M3):
- **NFE Steps**: ลดเป็น 16-24 เพื่อความเร็วที่ดีขึ้น
- **max_chars**: เริ่มต้นที่ 200-300 สำหรับความสมดุล
- **ตรวจสอบการใช้งาน GPU**: เปิด Activity Monitor → Window → GPU History

### คำแนะนำทั่วไป:
- สามารถตั้งค่า **"ตัวอักษรสูงสุดต่อส่วน"** หรือ `max_chars` เพื่อลดความผิดพลาดการอ่าน แต่ความเร็วในการสร้างจะช้าลง สามารถปรับลด NFE Step เพื่อเพิ่มความเร็วได้
- **อย่าลืมเว้นวรรคประโยค** เพื่อให้สามารถแบ่งส่วนในการสร้างได้
- สำหรับ `ref_text` หรือ ข้อความต้นฉบับ แนะนำให้ใช้เป็นภาษาไทยหรือคำอ่านภาษาไทยสำหรับเสียงภาษาอื่น เช่น Good Morning → กู้ดมอร์นิ่ง
- สำหรับ**เสียงต้นแบบ** ควรใช้ความยาวไม่เกิน 8 วินาที ถ้าเป็นไปได้ห้ามมีเสียงรบกวน
- สามารถปรับลดความเร็ว เพื่อให้การอ่านคำดีขึ้นได้ เช่น ความเร็ว 0.8-0.9 เพื่อลดการอ่านผิดหรือคำขาดหาย แต่ลดมากไปอาจมีเสียงต้นฉบับแทรกเข้ามา

---

## ตัวอย่าง WebUI

### Text To Speech
[Screenshot or demo]

### Multi Speech
[Screenshot or demo]

---

## ฝึกอบรม และ Finetune

### ใช้งานบน Google Colab
[Google Colab Finetune Notebook](link-to-your-colab)

### ติดตั้งสำหรับ Finetune บน macOS

```bash
cd F5-TTS-THAI

# ติดตั้งแบบ editable mode
pip install -e .
```

### เปิด Gradio Interface

```bash
f5-tts_finetune-gradio
```

---

## ตัวอย่างเสียง

### เสียงต้นฉบับ
**ข้อความ**: ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น

[ref_gen_1.mov]

### เสียงที่สร้าง 1 (ข้อความเดียวกัน)
**ข้อความ**: ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น

[tts_gen_1.mov]

### เสียงที่สร้าง 2 (ข้อความใหม่)
**ข้อความ**: ฉันชอบฟังเพลงขณะขับรถ เพราะช่วยให้รู้สึกผ่อนคลาย

[tts_gen_2.mov]

---

## การแก้ปัญหา (Troubleshooting)

### ถ้าไม่พบคำสั่ง `f5-tts_webui`
```bash
# ลองรันโดยตรงด้วย Python
python src/f5_tts/f5_tts_webui.py
```

### ถ้า MPS ไม่พร้อมใช้งาน
```bash
# ตรวจสอบเวอร์ชัน macOS (ต้องการ macOS 12.3+)
sw_vers

# อัพเดท PyTorch
pip install --upgrade torch torchaudio
```

### ถ้ามีปัญหาการ import module
```bash
# ติดตั้งใหม่ในโหมด editable
pip install -e .
```

---

## สคริปต์การติดตั้งอัตโนมัติ

สร้างไฟล์ `install_macos.sh`:

```bash
#!/bin/bash

echo "Installing F5-TTS-THAI for macOS..."

# ติดตั้ง prerequisites
echo "Installing prerequisites..."
brew install python@3.10 espeak-ng

# Clone repository
echo "Cloning repository..."
git clone https://github.com/VYNCX/F5-TTS-THAI.git
cd F5-TTS-THAI

# สร้างและเปิดใช้งาน venv
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ติดตั้ง packages
echo "Installing packages..."
pip install --upgrade pip
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git
pip install torch torchaudio

# ตรวจสอบการติดตั้ง
echo "Verifying installation..."
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

echo ""
echo "✅ Installation complete!"
echo "To start using F5-TTS-THAI:"
echo "1. Run: source venv/bin/activate"
echo "2. Run: f5-tts_webui"
echo "3. Open browser: http://127.0.0.1:7860"
```

รันสคริปต์:
```bash
chmod +x install_macos.sh
./install_macos.sh
```

---

## ข้อมูลเพิ่มเติม

### ความแตกต่างจาก Windows Version:
- ใช้ **MPS (Metal Performance Shaders)** แทน CUDA
- ไม่มีไฟล์ `.bat` - ใช้คำสั่ง shell script แทน
- Activation: `source venv/bin/activate` แทน `call venv/scripts/activate`

### ประสิทธิภาพที่คาดหวัง:
- **Apple Silicon (M1/M2/M3)**: ประสิทธิภาพดีด้วย MPS acceleration
- **Intel Mac**: ช้ากว่า เนื่องจากใช้ CPU เท่านั้น

---

## อ้างอิง

- [F5-TTS Original Repository](https://github.com/SWivid/F5-TTS)
- [Project Repository](https://github.com/VYNCX/F5-TTS-THAI)

---

## License

[Your License Information]

---

## ติดต่อและสนับสนุน

[Your Contact Information]










