import React, { useState } from 'react';
import styled from 'styled-components';
import { ApiService } from '../services/api';
import { TransactionRequest, FraudDetectionResult } from '../types';
import { toast } from 'react-toastify';

const Container = styled.div`
  padding: 2rem;
  max-width: 800px;
  margin: 0 auto;
`;

const Title = styled.h1`
  color: ${props => props.theme.colors.text.primary};
  margin-bottom: 1rem;
`;

const Form = styled.form`
  background: ${props => props.theme.colors.surface};
  padding: 2rem;
  border-radius: ${props => props.theme.borderRadius};
  box-shadow: ${props => props.theme.shadows.md};
  margin-bottom: 2rem;
`;

const FormGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
`;

const Label = styled.label`
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: ${props => props.theme.colors.text.primary};
`;

const Input = styled.input`
  padding: 0.75rem;
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius};
  font-size: 1rem;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.primary}20;
  }
`;

const Select = styled.select`
  padding: 0.75rem;
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius};
  font-size: 1rem;
  background: white;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.primary}20;
  }
`;

const Button = styled.button`
  background: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  padding: 0.75rem 2rem;
  border-radius: ${props => props.theme.borderRadius};
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
  
  &:hover {
    background: ${props => props.theme.colors.secondary};
  }
  
  &:disabled {
    background: ${props => props.theme.colors.text.disabled};
    cursor: not-allowed;
  }
`;

const ResultCard = styled.div<{ isfraud: boolean }>`
  background: ${props => props.isfraud ? '#fef2f2' : '#f0fdf4'};
  border: 1px solid ${props => props.isfraud ? '#fecaca' : '#bbf7d0'};
  border-radius: ${props => props.theme.borderRadius};
  padding: 1.5rem;
  margin-top: 1rem;
`;

const ResultTitle = styled.h3<{ isfraud: boolean }>`
  color: ${props => props.isfraud ? '#dc2626' : '#16a34a'};
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const ResultGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
`;

const ResultItem = styled.div`
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
  border-bottom: 1px solid #e5e7eb;
  
  &:last-child {
    border-bottom: none;
  }
`;

const TransactionTester: React.FC = () => {
  const [formData, setFormData] = useState<Partial<TransactionRequest>>({
    user_id: '',
    merchant_id: '',
    amount: 0,
    currency: 'USD',
    transaction_type: 'purchase',
    payment_method: 'credit_card',
    country: 'US',
  });
  
  const [result, setResult] = useState<FraudDetectionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.user_id || !formData.merchant_id || !formData.amount) {
      toast.error('Please fill in all required fields');
      return;
    }
    
    setLoading(true);
    try {
      const response = await ApiService.checkTransaction(formData as TransactionRequest);
      setResult(response);
      toast.success('Transaction analyzed successfully');
    } catch (error: any) {
      toast.error(error.message || 'Failed to analyze transaction');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: keyof TransactionRequest, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const generateSampleTransaction = () => {
    const samples = [
      {
        user_id: 'user_12345',
        merchant_id: 'Amazon',
        amount: 99.99,
        country: 'US',
        payment_method: 'credit_card' as const,
      },
      {
        user_id: 'user_67890',
        merchant_id: 'Suspicious Store',
        amount: 5000,
        country: 'CN',
        payment_method: 'cryptocurrency' as const,
      },
      {
        user_id: 'user_11111',
        merchant_id: 'Starbucks',
        amount: 4.50,
        country: 'US',
        payment_method: 'digital_wallet' as const,
      },
    ];
    
    const sample = samples[Math.floor(Math.random() * samples.length)];
    setFormData(prev => ({ ...prev, ...sample }));
  };

  return (
    <Container>
      <Title>🧪 Transaction Tester</Title>
      <p style={{ marginBottom: '2rem', color: '#64748b' }}>
        Test the fraud detection system by submitting sample transactions
      </p>

      <Form onSubmit={handleSubmit}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h3>Transaction Details</h3>
          <Button type="button" onClick={generateSampleTransaction}>
            Generate Sample
          </Button>
        </div>

        <FormGrid>
          <FormGroup>
            <Label>User ID *</Label>
            <Input
              value={formData.user_id || ''}
              onChange={(e) => handleInputChange('user_id', e.target.value)}
              placeholder="user_12345"
              required
            />
          </FormGroup>

          <FormGroup>
            <Label>Merchant ID *</Label>
            <Input
              value={formData.merchant_id || ''}
              onChange={(e) => handleInputChange('merchant_id', e.target.value)}
              placeholder="Amazon"
              required
            />
          </FormGroup>

          <FormGroup>
            <Label>Amount *</Label>
            <Input
              type="number"
              step="0.01"
              value={formData.amount || ''}
              onChange={(e) => handleInputChange('amount', parseFloat(e.target.value))}
              placeholder="99.99"
              required
            />
          </FormGroup>

          <FormGroup>
            <Label>Currency</Label>
            <Select
              value={formData.currency || 'USD'}
              onChange={(e) => handleInputChange('currency', e.target.value)}
            >
              <option value="USD">USD</option>
              <option value="EUR">EUR</option>
              <option value="GBP">GBP</option>
            </Select>
          </FormGroup>

          <FormGroup>
            <Label>Transaction Type</Label>
            <Select
              value={formData.transaction_type || 'purchase'}
              onChange={(e) => handleInputChange('transaction_type', e.target.value)}
            >
              <option value="purchase">Purchase</option>
              <option value="withdrawal">Withdrawal</option>
              <option value="transfer">Transfer</option>
              <option value="deposit">Deposit</option>
              <option value="refund">Refund</option>
            </Select>
          </FormGroup>

          <FormGroup>
            <Label>Payment Method</Label>
            <Select
              value={formData.payment_method || 'credit_card'}
              onChange={(e) => handleInputChange('payment_method', e.target.value)}
            >
              <option value="credit_card">Credit Card</option>
              <option value="debit_card">Debit Card</option>
              <option value="bank_transfer">Bank Transfer</option>
              <option value="digital_wallet">Digital Wallet</option>
              <option value="cryptocurrency">Cryptocurrency</option>
              <option value="cash">Cash</option>
            </Select>
          </FormGroup>

          <FormGroup>
            <Label>Country</Label>
            <Input
              value={formData.country || ''}
              onChange={(e) => handleInputChange('country', e.target.value)}
              placeholder="US"
            />
          </FormGroup>

          <FormGroup>
            <Label>IP Address</Label>
            <Input
              value={formData.ip_address || ''}
              onChange={(e) => handleInputChange('ip_address', e.target.value)}
              placeholder="192.168.1.1"
            />
          </FormGroup>
        </FormGrid>

        <Button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Transaction'}
        </Button>
      </Form>

      {result && (
        <ResultCard isfraud={result.is_fraud}>
          <ResultTitle isfraud={result.is_fraud}>
            {result.is_fraud ? '🚨 FRAUD DETECTED' : '✅ LEGITIMATE TRANSACTION'}
          </ResultTitle>
          
          <ResultGrid>
            <ResultItem>
              <span>Fraud Score:</span>
              <strong>{(result.fraud_score * 100).toFixed(1)}%</strong>
            </ResultItem>
            <ResultItem>
              <span>Confidence:</span>
              <strong>{(result.confidence * 100).toFixed(1)}%</strong>
            </ResultItem>
            <ResultItem>
              <span>Model Version:</span>
              <strong>{result.model_version}</strong>
            </ResultItem>
            <ResultItem>
              <span>Processing Time:</span>
              <strong>{result.processing_time_ms.toFixed(1)}ms</strong>
            </ResultItem>
          </ResultGrid>

          {result.fraud_reasons.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <strong>Fraud Indicators:</strong>
              <ul style={{ marginTop: '0.5rem' }}>
                {result.fraud_reasons.map((reason, index) => (
                  <li key={index} style={{ marginLeft: '1rem' }}>{reason}</li>
                ))}
              </ul>
            </div>
          )}
        </ResultCard>
      )}
    </Container>
  );
};

export default TransactionTester;