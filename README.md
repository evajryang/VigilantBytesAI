# VigilantBytes AI - Real-Time Fraud Detection System

VigilantBytes AI is a comprehensive real-time fraud detection system that uses machine learning algorithms to identify potentially fraudulent transactions as they occur.

## Features

- **Real-time Transaction Processing**: Process transactions instantly with sub-second response times
- **Machine Learning Detection**: Multiple ML models including Isolation Forest and Neural Networks
- **Interactive Dashboard**: Real-time monitoring dashboard with live alerts
- **API-First Design**: RESTful API for easy integration with existing systems
- **Scalable Architecture**: Built with FastAPI, Redis, and containerized deployment
- **Historical Analysis**: Store and analyze transaction patterns over time

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   ML Engine     │
│   (React)       │◄──►│   (FastAPI)      │◄──►│   (Scikit/TF)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Database       │    │   Redis Cache   │
                       │   (SQLite/PG)    │    │   (Real-time)   │
                       └──────────────────┘    └─────────────────┘
```

## Quick Start

### Using Docker (Recommended)
```bash
docker-compose up -d
```

### Manual Setup
```bash
# Backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm start
```

## API Endpoints

- `POST /api/transactions/check` - Submit transaction for fraud detection
- `GET /api/transactions/history` - Get transaction history
- `GET /api/analytics/stats` - Get fraud detection statistics
- `WebSocket /ws/alerts` - Real-time fraud alerts

## Technology Stack

- **Backend**: FastAPI, Python 3.9+
- **ML**: Scikit-learn, TensorFlow, Pandas, NumPy
- **Database**: SQLite (dev), PostgreSQL (prod)
- **Cache**: Redis
- **Frontend**: React, TypeScript, Chart.js
- **Deployment**: Docker, Docker Compose

## License

MIT License - see LICENSE file for details
