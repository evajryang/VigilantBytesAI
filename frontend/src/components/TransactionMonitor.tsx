import React from 'react';
import styled from 'styled-components';

const Container = styled.div`
  padding: 2rem;
  text-align: center;
`;

const TransactionMonitor: React.FC = () => {
  return (
    <Container>
      <h1>🔍 Transaction Monitor</h1>
      <p>Real-time transaction monitoring and analysis</p>
      <p style={{ color: '#64748b' }}>This component will show live transaction feeds and filtering options.</p>
    </Container>
  );
};

export default TransactionMonitor;