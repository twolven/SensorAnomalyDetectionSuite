// src/components/SettingsPanel.js
export const SettingsPanel = ({ settings, onUpdate }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [localSettings, setLocalSettings] = useState(settings);
  
    const handleSave = () => {
      onUpdate(localSettings);
      setIsOpen(false);
    };
  
    return (
      <div className="relative">
        <button
          onClick={() => setIsOpen(true)}
          className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>
  
        {isOpen && (
          <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-lg p-4">
            <h3 className="font-medium mb-4">Settings</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Sample Rate (Hz)
                </label>
                <input
                  type="number"
                  value={localSettings.sampleRate}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    sampleRate: parseInt(e.target.value)
                  }))}
                  className="w-full p-2 border rounded"
                />
              </div>
  
              <div>
                <label className="block text-sm font-medium mb-1">
                  Buffer Size
                </label>
                <input
                  type="number"
                  value={localSettings.bufferSize}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    bufferSize: parseInt(e.target.value)
                  }))}
                  className="w-full p-2 border rounded"
                />
              </div>
  
              <div className="flex justify-end space-x-2">
                <button
                  onClick={() => setIsOpen(false)}
                  className="px-3 py-1 rounded bg-gray-100 hover:bg-gray-200"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  className="px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-700"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };