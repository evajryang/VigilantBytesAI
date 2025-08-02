import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { toast } from 'react-toastify';
import { WSMessage, RealTimeAlert, AnalyticsStats } from '../types';

interface WebSocketContextType {
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastMessage: WSMessage | null;
  connect: () => void;
  disconnect: () => void;
  sendMessage: (message: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: React.ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [shouldReconnect, setShouldReconnect] = useState(true);

  const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws/alerts';
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000;

  const connect = useCallback(() => {
    if (socket?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionStatus('connecting');
    
    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setSocket(ws);
        setIsConnected(true);
        setConnectionStatus('connected');
        setReconnectAttempts(0);
        
        toast.success('Real-time monitoring enabled', {
          position: 'top-right',
          autoClose: 3000,
        });
      };

      ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data);
          setLastMessage(message);
          
          // Handle different message types
          switch (message.type) {
            case 'fraud_alert':
              handleFraudAlert(message.data);
              break;
            case 'stats_update':
              handleStatsUpdate(message.data);
              break;
            case 'heartbeat':
              // Just keep the connection alive
              break;
            default:
              console.log('Unknown message type:', message.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
        toast.error('WebSocket connection error', {
          position: 'top-right',
          autoClose: 5000,
        });
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setSocket(null);
        setIsConnected(false);
        setConnectionStatus('disconnected');

        // Attempt to reconnect if not manually closed
        if (shouldReconnect && reconnectAttempts < maxReconnectAttempts) {
          setTimeout(() => {
            setReconnectAttempts(prev => prev + 1);
            connect();
          }, reconnectDelay);
        }
      };

    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      setConnectionStatus('error');
    }
  }, [socket, wsUrl, reconnectAttempts, shouldReconnect]);

  const disconnect = useCallback(() => {
    setShouldReconnect(false);
    if (socket) {
      socket.close();
    }
  }, [socket]);

  const sendMessage = useCallback((message: any) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, [socket]);

  const handleFraudAlert = (alertData: RealTimeAlert) => {
    const severityEmoji = {
      low: '🟡',
      medium: '🟠', 
      high: '🔴',
      critical: '🚨'
    };

    toast.error(
      <div>
        <strong>
          {severityEmoji[alertData.severity]} Fraud Alert
        </strong>
        <br />
        {alertData.message}
        <br />
        <small>
          Amount: ${alertData.amount.toLocaleString()} | 
          Score: {(alertData.fraud_score * 100).toFixed(1)}%
        </small>
      </div>,
      {
        position: 'top-right',
        autoClose: alertData.requires_action ? false : 8000,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      }
    );

    // Play alert sound (optional)
    if (alertData.severity === 'high' || alertData.severity === 'critical') {
      try {
        const audio = new Audio('/alert-sound.mp3');
        audio.play().catch(() => {
          // Ignore audio errors (user hasn't interacted with page yet)
        });
      } catch (error) {
        // Audio not available
      }
    }
  };

  const handleStatsUpdate = (statsData: Partial<AnalyticsStats>) => {
    // Update global stats or trigger refresh
    // This could be connected to a global state management solution
    console.log('Stats updated:', statsData);
  };

  // Auto-connect on mount
  useEffect(() => {
    connect();

    return () => {
      setShouldReconnect(false);
      if (socket) {
        socket.close();
      }
    };
  }, []);

  // Reconnect when window gains focus
  useEffect(() => {
    const handleFocus = () => {
      if (!isConnected && shouldReconnect) {
        connect();
      }
    };

    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, [isConnected, shouldReconnect, connect]);

  const value: WebSocketContextType = {
    isConnected,
    connectionStatus,
    lastMessage,
    connect,
    disconnect,
    sendMessage,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};