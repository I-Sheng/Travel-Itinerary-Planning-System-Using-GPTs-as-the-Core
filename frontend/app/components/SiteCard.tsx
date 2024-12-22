import React from "react";
import Image from "next/image";
import Link from "next/link";
// import Link from "next/link";

interface CardProps {
  title: string;
  imageSrc: string;
  alt: string;
  isFood: boolean;
}

const SiteCard: React.FC<CardProps> = ({ title, imageSrc, alt, isFood }) => {
  return (
    <>
      <div
        className={`flex flex-col items-center p-4 rounded-lg text-black shadow-md h-full w-full ${
          isFood ? "bg-gray-200 " : "bg-white"
        }`}
      >
        <h2 className="text-lg font-bold mt-2 ml-2 content-center">{title}</h2>
        <Image
          src={imageSrc}
          alt={alt}
          width={300}
          height={125}
          loading="lazy"
          sizes="(max-width: 600px) 100vw, (max-width: 1200px) 50vw, 600px"
          className="rounded-lg object-cover  sm:max-h-[80%]   max-w-[100%]"
        />
        <div className="flex flex-col sm:flex-row items-center sm:space-x-4 mt-4 space-y-2 sm:space-y-0">
          {/* <Link href={`/sites/${title.replace(/\s/g, "_")}`} passHref> */}
          <Link href={`/sites/${encodeURIComponent(title)}`} passHref>
            <button
              type="submit"
              className="font-semibold  hover:text-blue-500"
            >
              Learn more...
            </button>
          </Link>
        </div>
      </div>
    </>
  );
};

export default SiteCard;
