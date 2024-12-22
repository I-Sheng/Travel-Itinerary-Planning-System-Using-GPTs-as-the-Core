import React from "react";
import Image from "next/image";
import Link from "next/link";

const Header: React.FC = () => {
  return (
    <header className="flex items-center justify-between bg-white shadow-md py-4 px-8">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <Link href="/" passHref>
          <Image src="/favicon.ico" alt="ITravel Logo" width={24} height={24} />
        </Link>
      </div>

      {/* Brand Name */}
      <div className="absolute left-1/2 transform -translate-x-1/2 h-10">
        <Link href="/" passHref>
          <div className="text-3xl font-bold text-blue-600 hover:text-blue-800 transition-colors">
            ITravel
          </div>
        </Link>
      </div>

      {/* Navigation Links */}
      <nav className="flex gap-3">
        <Link href="/traveling" passHref>
          <div className="text-blue-600 hover:text-blue-800 font-medium text-sm transition-colors">
            Planner
          </div>
        </Link>
        <Link href="/sites" passHref>
          <div className="text-blue-600 hover:text-blue-800 font-medium text-sm transition-colors">
            Sites
          </div>
        </Link>
      </nav>
    </header>
  );
};

export default Header;
