#!/bin/bash

echo "빌드 및 실행을 시작합니다..."

echo "백엔드(FastAPI) 세팅 중..."
cd backend
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/Scripts/activate
pip install -r requirements.txt > /dev/null 2>&1
echo "백엔드 서버를 8000 포트에서 실행합니다."
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

echo "프론트엔드(React) 세팅 중..."
cd frontend
npm install > /dev/null 2>&1
echo "프론트엔드 서버를 실행합니다."
npm run dev &
FRONTEND_PID=$!
cd ..

echo "모든 서버가 실행되었습니다!"
echo "- Frontend: http://localhost:3000"
echo "- Backend API Docs: http://localhost:8000/docs"
echo "종료하려면 [CTRL+C]를 누르세요."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '서버가 종료되었습니다.'" EXIT

wait