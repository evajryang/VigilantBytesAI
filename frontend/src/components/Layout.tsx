import React from 'react';
import { Outlet } from 'react-router-dom';
import styled from 'styled-components';

const LayoutContainer = styled.div`
  display: flex;
  min-height: 100vh;
  background: ${props => props.theme.colors.background};
`;

const MainContent = styled.main`
  flex: 1;
  overflow-y: auto;
`;

const Layout: React.FC = () => {
  return (
    <LayoutContainer>
      <MainContent>
        <Outlet />
      </MainContent>
    </LayoutContainer>
  );
};

export default Layout;