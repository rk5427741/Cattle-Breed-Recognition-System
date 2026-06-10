/**
 * Backend API base URLs.
 * - Dev (Vite): `/api` is proxied to http://localhost:8000 (see vite.config.js)
 * - Override: set VITE_API_URL in frontend/.env
 * - Fallback: direct URL if proxy/backend is on another host
 */
export const API_URL = import.meta.env.VITE_API_URL || '/api'

export const API_URL_DIRECT =
  import.meta.env.VITE_API_DIRECT_URL || 'http://127.0.0.1:8000'

/**
 * GET with automatic fallback when Vite proxy is unavailable.
 */
export async function apiGet(path, config = {}) {
  const normalized = path.startsWith('/') ? path : `/${path}`
  const bases = import.meta.env.VITE_API_URL
    ? [import.meta.env.VITE_API_URL.replace(/\/$/, '')]
    : [API_URL.replace(/\/$/, ''), API_URL_DIRECT.replace(/\/$/, '')]

  let lastError = null
  for (const base of bases) {
    try {
      const axios = (await import('axios')).default
      const response = await axios.get(`${base}${normalized}`, {
        timeout: 15000,
        ...config,
      })
      return response
    } catch (error) {
      lastError = error
      if (import.meta.env.DEV) {
        console.warn(`API request failed for ${base}${normalized}:`, error.message)
      }
    }
  }
  throw lastError
}

/**
 * POST multipart (e.g. image upload) with same fallback behavior.
 */
export async function apiPostForm(path, formData, config = {}) {
  const normalized = path.startsWith('/') ? path : `/${path}`
  const bases = import.meta.env.VITE_API_URL
    ? [import.meta.env.VITE_API_URL.replace(/\/$/, '')]
    : [API_URL.replace(/\/$/, ''), API_URL_DIRECT.replace(/\/$/, '')]

  let lastError = null
  for (const base of bases) {
    try {
      const axios = (await import('axios')).default
      const response = await axios.post(`${base}${normalized}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
        ...config,
      })
      return response
    } catch (error) {
      lastError = error
      if (import.meta.env.DEV) {
        console.warn(`API POST failed for ${base}${normalized}:`, error.message)
      }
    }
  }
  throw lastError
}
