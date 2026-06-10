import { useState } from 'react'
import { apiPostForm } from '../api'
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png']
const INVALID_MSG =
  'Invalid image. Please upload an image of Indian cattle or buffalo.'

function Home() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const validateClientFile = (file) => {
    if (!file) return 'Please select an image first.'
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (!['jpg', 'jpeg', 'png'].includes(ext || '')) {
      return 'Only JPG, JPEG, and PNG files are supported.'
    }
    if (!ALLOWED_TYPES.includes(file.type) && file.type !== '') {
      return 'Only JPG, JPEG, and PNG files are supported.'
    }
    if (file.size > 16 * 1024 * 1024) {
      return 'File size must be under 16MB.'
    }
    return null
  }

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (!file) return

    const validationError = validateClientFile(file)
    if (validationError) {
      setError(validationError)
      setSelectedFile(null)
      setPreview(null)
      return
    }

    setSelectedFile(file)
    setError(null)
    setResult(null)

    const reader = new FileReader()
    reader.onloadend = () => setPreview(reader.result)
    reader.readAsDataURL(file)
  }

  const handlePredict = async () => {
    const validationError = validateClientFile(selectedFile)
    if (validationError) {
      setError(validationError)
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await apiPostForm('/predict', formData)

      if (response.data.rejected) {
        setError(response.data.message || INVALID_MSG)
        setResult(null)
      } else if (response.data.success) {
        setResult(response.data)
      } else {
        setError('Prediction failed. Please try again.')
      }
    } catch (err) {
      const data = err.response?.data
      if (data?.rejected || err.response?.status === 422) {
        setError(data?.message || INVALID_MSG)
      } else {
        setError(
          data?.detail ||
            data?.message ||
            'Failed to connect to the server. Make sure the backend is running on port 8000.'
        )
      }
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">
          Cattle Breed Recognition System
        </h1>
        <p className="text-gray-600">
          Upload a clear photo of Indian cattle or buffalo. Unrelated images are
          rejected automatically.
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-xl p-8">
        <div className="mb-8">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Image (JPG, JPEG, PNG)
          </label>
          <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md hover:border-indigo-500 transition-colors">
            <div className="space-y-1 text-center">
              <div className="flex text-sm text-gray-600 justify-center">
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer font-medium text-indigo-600 hover:text-indigo-500"
                >
                  Upload a file
                  <input
                    id="file-upload"
                    type="file"
                    className="sr-only"
                    accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                    onChange={handleFileSelect}
                  />
                </label>
              </div>
              <p className="text-xs text-gray-500">JPG, JPEG, PNG — max 16MB</p>
            </div>
          </div>
        </div>

        {preview && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Preview</h3>
            <img
              src={preview}
              alt="Preview"
              className="w-full h-64 object-contain bg-gray-50 rounded-lg border"
            />
          </div>
        )}

        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}

        <div className="flex gap-4 justify-center">
          <button
            onClick={handlePredict}
            disabled={!selectedFile || loading}
            className="px-8 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <svg
                  className="animate-spin h-5 w-5"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Analysing image...
              </>
            ) : (
              'Recognise Breed'
            )}
          </button>
          {(selectedFile || result) && (
            <button
              onClick={handleReset}
              className="px-8 py-3 bg-gray-200 text-gray-700 rounded-lg font-semibold hover:bg-gray-300"
            >
              Reset
            </button>
          )}
        </div>

        {result && (
          <div className="mt-8 bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-6 border-2 border-green-200">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">
              Prediction Result
            </h2>

            <div className="bg-white rounded-lg p-6 shadow-md">
              <div className="text-center mb-4">
                <div className="text-4xl font-bold text-indigo-600 mb-2">
                  {result.breed}
                </div>
                <div className="text-sm text-gray-500 mb-4">
                  Confidence:{' '}
                  <span className="font-semibold text-indigo-600">
                    {result.confidence}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4 mb-6">
                  <div
                    className={`h-4 rounded-full transition-all duration-700 ${
                      result.confidence >= 80
                        ? 'bg-green-500'
                        : result.confidence >= 60
                          ? 'bg-yellow-500'
                          : 'bg-orange-500'
                    }`}
                    style={{ width: `${Math.min(result.confidence, 100)}%` }}
                  />
                </div>
              </div>

              {result.top_predictions?.length > 0 && (
                <div className="mb-4 border-t pt-4">
                  <h3 className="font-medium text-gray-700 mb-2">
                    Top 3 probable breeds
                  </h3>
                  <ul className="space-y-2">
                    {result.top_predictions.map((row, idx) => (
                      <li
                        key={`${row.breed}-${idx}`}
                        className="flex justify-between text-sm bg-gray-50 px-3 py-2 rounded"
                      >
                        <span>
                          {idx + 1}. {row.breed}
                        </span>
                        <span className="font-semibold text-indigo-600">
                          {row.confidence}%
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {result.details && (
                <div className="border-t pt-4 space-y-2 text-sm">
                  <div className="flex justify-between gap-4">
                    <span className="text-gray-600 font-medium">Type</span>
                    <span className="text-gray-800 text-right">{result.details.type}</span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-gray-600 font-medium">Milk yield</span>
                    <span className="text-gray-800 text-right">{result.details.milk_yield}</span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-gray-600 font-medium">Region</span>
                    <span className="text-gray-800 text-right">{result.details.region}</span>
                  </div>
                  {result.details.characteristics && (
                    <p className="text-gray-700 pt-2 border-t">
                      {result.details.characteristics}
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Home
