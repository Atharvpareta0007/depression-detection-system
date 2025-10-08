import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain, Home, Info, Sparkles } from 'lucide-react';

function Navbar() {
  const location = useLocation();
  
  const isActive = (path) => location.pathname === path;
  
  return (
    <motion.nav 
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="relative z-20 mx-4 mt-4 mb-8"
    >
      <div className="glass-card-modern py-4 px-6 rounded-2xl backdrop-blur-xl bg-white/5 border border-white/10 shadow-2xl">
        <div className="container mx-auto flex items-center justify-between">
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="relative">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-0 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full blur-md opacity-30"
                />
                <Brain className="relative w-10 h-10 text-transparent bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text drop-shadow-lg" />
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-2xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
                  Depression Detection AI
                </span>
                <Sparkles className="w-5 h-5 text-cyan-400 animate-pulse" />
              </div>
            </Link>
          </motion.div>
          
          <div className="flex space-x-2">
            <motion.div
              whileHover={{ scale: 1.05, rotateY: 5 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link
                to="/"
                className={`relative flex items-center space-x-2 px-6 py-3 rounded-xl transition-all duration-300 ${
                  isActive('/') 
                    ? 'bg-gradient-to-r from-purple-500/20 to-cyan-500/20 text-white shadow-lg shadow-purple-500/20 border border-purple-500/30' 
                    : 'hover:bg-white/10 hover:shadow-lg hover:shadow-white/10 text-gray-300 hover:text-white'
                }`}
              >
                {isActive('/') && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-cyan-500/20 rounded-xl border border-purple-500/30"
                    initial={false}
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
                <Home className="relative w-5 h-5" />
                <span className="relative font-medium">Home</span>
              </Link>
            </motion.div>
            
            <motion.div
              whileHover={{ scale: 1.05, rotateY: 5 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link
                to="/about"
                className={`relative flex items-center space-x-2 px-6 py-3 rounded-xl transition-all duration-300 ${
                  isActive('/about') 
                    ? 'bg-gradient-to-r from-purple-500/20 to-cyan-500/20 text-white shadow-lg shadow-purple-500/20 border border-purple-500/30' 
                    : 'hover:bg-white/10 hover:shadow-lg hover:shadow-white/10 text-gray-300 hover:text-white'
                }`}
              >
                {isActive('/about') && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-cyan-500/20 rounded-xl border border-purple-500/30"
                    initial={false}
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
                <Info className="relative w-5 h-5" />
                <span className="relative font-medium">About</span>
              </Link>
            </motion.div>
          </div>
        </div>
      </div>
    </motion.nav>
  );
}

export default Navbar;
