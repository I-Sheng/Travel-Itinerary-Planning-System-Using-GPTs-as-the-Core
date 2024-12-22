import React, { useState } from "react";
import Switch from "@/app/components/Switch";

interface CardProps {
  title: string;
  openingTime: string;
  stay_time: number | null;
  isFood: boolean;
  onToggle: (value: string) => void;
}

const Card: React.FC<CardProps> = ({
  title,
  openingTime,
  stay_time,
  isFood,
  onToggle,
}) => {
  function transferTime(time: number) {
    const hours = Math.floor(time / 60);
    const minutes = time % 60;
    if (hours === 0) {
      return `${minutes} 分鐘`;
    }
    if (minutes === 0) {
      return `${hours} 小時`;
    }
    return `${hours} 小時  ${minutes} 分鐘`;
  }
  const [isOn, setIsOn] = useState<boolean>(false);

  const handleToggle = () => {
    setIsOn(!isOn);
    onToggle(title);
  };

  const splitText = (text: string) => {
    const s: string[] = text.split(" ");
    return s.slice(1).join(" ");
  };
  return (
    <>
      <div
        className={`flex flex-col items-center p-4 rounded-lg shadow-md text-black h-full w-full ${
          isFood ? "bg-pink-200" : "bg-purple-200"
        }
      `}
      >
        <h2 className="text-lg font-bold mt-2 ml-2 content-center">{title}</h2>
        <p>{`週日開放時間: ${splitText(openingTime)}`}</p>
        {stay_time !== null && <p>預估停留時間: {transferTime(stay_time)}</p>}
        <div className="flex flex-col sm:flex-row items-center sm:space-x-4 mt-4 space-y-2 sm:space-y-0">
          <Switch
            isOn={isOn}
            handleToggle={handleToggle}
            id={`switch-${title}`}
          />
          <button
            type="button"
            className="font-semibold  hover:text-blue-500"
            onClick={() =>
              window.open(`/sites/${encodeURIComponent(title)}`, "_blank")
            }
          >
            Learn more...
          </button>
        </div>
      </div>
    </>
  );
};

export default Card;
