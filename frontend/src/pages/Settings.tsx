import { Sidebar } from '../components/layout/Sidebar';

const Settings = () => {
  return (
    <div className="flex min-h-screen bg-nebula-bg text-nebula-text-main">
      <Sidebar />
      <main className="flex-1 ml-72 p-8">
        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-nebula-primary to-nebula-secondary mb-8">
            Settings
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Appearance */}
            <section className="bg-nebula-surface/50 backdrop-blur border border-nebula-border rounded-2xl p-6">
                <h2 className="text-xl font-semibold mb-4 text-nebula-text-main">Appearance</h2>
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <span className="text-nebula-text-muted">High Quality Rendering</span>
                        <div className="w-12 h-6 bg-nebula-primary/20 rounded-full relative cursor-pointer">
                             <div className="absolute right-1 top-1 w-4 h-4 bg-nebula-primary rounded-full shadow-glow"></div>
                        </div>
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="text-nebula-text-muted">Show Trajectories</span>
                        <div className="w-12 h-6 bg-nebula-surfaceHighlight rounded-full relative cursor-pointer border border-nebula-border">
                             <div className="absolute left-1 top-1 w-4 h-4 bg-nebula-text-dim rounded-full"></div>
                        </div>
                    </div>
                </div>
            </section>

             {/* API Configuration */}
             <section className="bg-nebula-surface/50 backdrop-blur border border-nebula-border rounded-2xl p-6">
                <h2 className="text-xl font-semibold mb-4 text-nebula-text-main">Connection</h2>
                <div className="space-y-4">
                    <div>
                        <label className="block text-sm text-nebula-text-muted mb-1">API Endpoint</label>
                        <input
                            type="text"
                            value="http://localhost:8000/api"
                            readOnly
                            className="w-full bg-nebula-bg border border-nebula-border rounded-lg px-3 py-2 text-nebula-text-dim"
                        />
                    </div>
                    <div>
                         <label className="block text-sm text-nebula-text-muted mb-1">GitHub Token Status</label>
                         <div className="flex items-center gap-2 text-green-400 text-sm">
                             <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                             Connected
                         </div>
                    </div>
                </div>
            </section>
        </div>
      </main>
    </div>
  );
};

export default Settings;
