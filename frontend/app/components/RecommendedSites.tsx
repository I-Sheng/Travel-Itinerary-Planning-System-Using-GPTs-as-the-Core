"use client";

import React, { useState, useEffect } from "react";
import Card from "@/app/components/Cards";
import data from "@/data/sitesData.json";
import ScheduleSitesComponent from "./ScheduleSites";
import Image from "next/image";

interface RecommendedSitesProps {
  day: number;
  preference: string;
  onSubmit: () => void;
}

const RecommendedSitesComponent: React.FC<RecommendedSitesProps> = ({
  day,
  preference,
  onSubmit,
}) => {
  const [recommendedSites, setRecommendedSites] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);
  const [submitted, setSubmitted] = useState<boolean>(false);
  const [isLoading, setLoading] = useState<boolean>(true);
  const toggleOption = (option: string) => {
    setSelectedOptions((prevSelectedOptions) => {
      const newSelectedOptions = prevSelectedOptions.includes(option)
        ? prevSelectedOptions.filter((opt) => opt !== option)
        : [...prevSelectedOptions, option];

      return newSelectedOptions;
    });
  };

  const handleSubmit = () => {
    setSubmitted(!submitted); // Set submitted to true to show confirmation
  };

  const returnSubmit = () => {
    handleSubmit();
    setSelectedOptions([]);
  };

  useEffect(() => {
    async function fetchRecommendedSites() {
      try {
        const response = await fetch("/api/recommend", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ day, preference }),
        });

        if (!response.ok) {
          throw new Error("Failed to fetch recommended sites");
        }

        const data = await response.json();
        setLoading(false);
        if (data.result && Array.isArray(data.result)) {
          setRecommendedSites(data.result);
        } else {
          throw new Error("Unexpected data structure");
        }
      } catch (error: unknown) {
        if (error instanceof Error) {
          setError(error.message);
        } else {
          setError("An unknown error occurred");
        }
      }
    }

    fetchRecommendedSites();
    setSelectedOptions([]);
  }, [day, preference]); // Dependencies to re-fetch when day or preference changes

  if (error) {
    return <p>Error: {error}</p>;
  }

  return (
    <>
      {submitted ? (
        <ScheduleSitesComponent
          day={day}
          sites={selectedOptions.join(", ")}
          start_time={480}
          end_time={1200}
          start_point="嘉義火車站"
          onSubmit={returnSubmit}
        />
      ) : (
        <div>
          {isLoading ? (
            <div className="flex flex-col items-center mt-[10%] justify-center gap-8 h-[100%]">
              <Image
                src="/loading.gif"
                alt="Waiting for AI generation..."
                width={65}
                height={65}
                className="rounded-lg object-cover"
                unoptimized
              />
              <p className="text-lg object-cover">Awaiting AI insight...</p>
            </div>
          ) : (
            <>
              <h2 className="text-2xl font-bold text-center mt-4">景點推薦</h2>
              <ul className="grid xl:grid-cols-4 lg:grid-cols-3 md:grid-cols-2 gap-8 mt-10 mb-20 mx-28">
                {recommendedSites.map((site: string, index) => (
                  <li key={index} className="flex py-1">
                    <Card
                      title={site}
                      openingTime={
                        data[site as keyof typeof data]["opening_hours"][6]
                      }
                      stay_time={
                        Math.ceil(
                          (data[site as keyof typeof data]["time_spent"][0] +
                            data[site as keyof typeof data]["time_spent"][1]) /
                            2 /
                            5
                        ) * 5
                      }
                      isFood={
                        data[site as keyof typeof data]["metadata"]["type"] ===
                        "food"
                      }
                      onToggle={toggleOption}
                    />
                  </li>
                ))}
              </ul>
              <div className="flex items-center justify-center gap-36 mt-5 mb-20">
                <form onSubmit={onSubmit}>
                  <button
                    type="submit"
                    className="bg-blue-600 text-white py-2 px-5 font-semibold rounded-lg shadow-md transition-transform transform hover:scale-105 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  >
                    上一頁
                  </button>
                </form>
                <form onSubmit={handleSubmit}>
                  <button
                    type="submit"
                    className="bg-green-600 text-white py-2 px-5 font-semibold rounded-lg shadow-md transition-transform transform hover:scale-105 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
                  >
                    旅程規劃
                  </button>
                </form>
                {/* <p>{selectedOptions.join(", ")}</p> */}
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
};

export default RecommendedSitesComponent;
