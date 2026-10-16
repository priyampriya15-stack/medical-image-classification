import type { User } from '../types';

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface AuthResponse {
  token: string;
  user: User;
}

const STORAGE_KEY_TOKEN = 'medai_auth_token';
const STORAGE_KEY_USER = 'medai_auth_user';

export const authService = {
  // Login with credentials
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    // Simulate network latency for realistic feel
    await new Promise((resolve) => setTimeout(resolve, 600));

    // Basic client validation
    const email = credentials.email.trim();
    if (!email || !credentials.password) {
      throw new Error('Please enter both email and password.');
    }

    if (!email.includes('@')) {
      throw new Error('Please enter a valid academic or professional email.');
    }

    if (credentials.password.length < 6) {
      throw new Error('Password must be at least 6 characters.');
    }

    // Determine user profile based on input or research defaults
    const userName = email.split('@')[0].replace(/[._-]/g, ' ');
    const formattedName = userName.charAt(0).toUpperCase() + userName.slice(1);

    const user: User = {
      id: 'usr-' + Math.random().toString(36).substring(2, 9),
      name: formattedName.length > 2 && formattedName.toLowerCase() !== 'researcher' ? formattedName : 'Priya M',
      email: email,
      role: 'Lead AI Research Engineer',
      institution: 'Medical Imaging Research Lab',
      avatarUrl: undefined,
    };

    const token = 'jwt-session-' + btoa(JSON.stringify({ sub: user.id, exp: Date.now() + 86400000 }));

    if (credentials.rememberMe) {
      localStorage.setItem(STORAGE_KEY_TOKEN, token);
      localStorage.setItem('medai_token', token);
      localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(user));
    } else {
      sessionStorage.setItem(STORAGE_KEY_TOKEN, token);
      sessionStorage.setItem('medai_token', token);
      sessionStorage.setItem(STORAGE_KEY_USER, JSON.stringify(user));
    }

    return { token, user };
  },

  // Logout
  async logout(): Promise<void> {
    localStorage.removeItem(STORAGE_KEY_TOKEN);
    localStorage.removeItem('medai_token');
    localStorage.removeItem(STORAGE_KEY_USER);
    sessionStorage.removeItem(STORAGE_KEY_TOKEN);
    sessionStorage.removeItem('medai_token');
    sessionStorage.removeItem(STORAGE_KEY_USER);
  },

  // Get current authenticated user from storage
  getCurrentUser(): User | null {
    try {
      const stored = localStorage.getItem(STORAGE_KEY_USER) || sessionStorage.getItem(STORAGE_KEY_USER);
      if (!stored) return null;
      return JSON.parse(stored) as User;
    } catch {
      return null;
    }
  },

  // Check if session token exists
  hasValidToken(): boolean {
    const token = localStorage.getItem(STORAGE_KEY_TOKEN) || sessionStorage.getItem(STORAGE_KEY_TOKEN);
    return Boolean(token);
  },
};
