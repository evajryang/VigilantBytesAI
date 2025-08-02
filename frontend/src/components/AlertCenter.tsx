import React from 'react';
import styled from 'styled-components';

const Container = styled.div`
  padding: 2rem;
  text-align: center;
`;

const AlertCenter: React.FC = () => {
  return (
    <Container>
      <h1>🚨 Alert Center</h1>
      <p>Fraud alert management and investigation tools</p>
      <p style={{ color: '#64748b' }}>This component will show active alerts and investigation workflows.</p>
    </Container>
  );
};

export default AlertCenter;