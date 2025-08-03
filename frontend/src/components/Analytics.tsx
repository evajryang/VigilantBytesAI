import React from 'react';
import styled from 'styled-components';

const Container = styled.div`
  padding: 2rem;
  text-align: center;
`;

const Analytics: React.FC = () => {
  return (
    <Container>
      <h1>📊 Analytics</h1>
      <p>Deep fraud pattern analysis and reporting</p>
      <p style={{ color: '#64748b' }}>This component will show charts, trends, and detailed analytics.</p>
    </Container>
  );
};

export default Analytics;