import { useState, useEffect } from 'react'
import { apiGet, API_URL, API_URL_DIRECT } from '../api'

function About() {
  const [breeds, setBreeds] = useState([])
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState(null)
  const [apiBaseUsed, setApiBaseUsed] = useState(null)

  useEffect(() => {
    fetchBreeds()
  }, [])

  const fetchBreeds = async () => {
    setLoading(true)
    setFetchError(null)
    try {
      const response = await apiGet('/breeds')
      if (response.data.success) {
        setBreeds(response.data.breeds)
        setApiBaseUsed(response.config?.baseURL || API_URL)
      } else {
        setFetchError('API returned an unexpected response.')
      }
    } catch (error) {
      console.error('Error fetching breeds:', error)
      const detail = error.response?.data?.detail
      const isNetwork =
        error.code === 'ERR_NETWORK' ||
        error.message?.includes('Network Error')
      setFetchError(
        detail ||
          (isNetwork
            ? `Cannot reach the backend. Start it with: cd backend && python main.py (expected at ${API_URL_DIRECT} or Vite proxy ${API_URL})`
            : 'Cannot load breed list. Check that the backend is running on port 8000.')
      )
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading breed information...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">
          About Cattle Breeds
        </h1>
        <p className="text-gray-600">
          Learn about different cattle breeds supported by our recognition system
        </p>
        {apiBaseUsed && breeds.length > 0 && (
          <p className="text-xs text-gray-400 mt-2">Connected to backend</p>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {breeds.map((breed, index) => (
          <div
            key={breed.key || index}
            className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-shadow"
          >
            <div className="bg-gradient-to-r from-indigo-500 to-purple-600 p-6">
              <h2 className="text-2xl font-bold text-white">{breed.name}</h2>
              <p className="text-indigo-100 text-sm mt-1">{breed.type}</p>
            </div>

            <div className="p-6 space-y-3">
              <div>
                <span className="text-gray-600 font-medium">Milk Yield:</span>
                <p className="text-gray-800 font-semibold">{breed.milk_yield}</p>
              </div>

              <div>
                <span className="text-gray-600 font-medium">Region:</span>
                <p className="text-gray-800">{breed.region}</p>
              </div>

              {breed.characteristics && (
                <div>
                  <span className="text-gray-600 font-medium">Characteristics:</span>
                  <p className="text-gray-800 text-sm mt-1">{breed.characteristics}</p>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {!loading && breeds.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-600">No breed information available.</p>
          <p className="text-red-600 text-sm mt-2 max-w-lg mx-auto">
            {fetchError ||
              'Make sure the backend API is running: cd backend && python main.py'}
          </p>
          <button
            type="button"
            onClick={fetchBreeds}
            className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
          >
            Retry
          </button>
        </div>
      )}
    </div>
  )
}

export default About
