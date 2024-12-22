import data from "@/data/sitesData.json";
import SiteCard from "../components/SiteCard";

interface SiteNameProps {
  name: string;
  metadata: { type: string };
}

const Site: React.FC = () => {
  return (
    <>
      <ul className="grid xl:grid-cols-4 lg:grid-cols-3 md:grid-cols-2 gap-8 mt-20 mb-20 mx-28">
        {(Object.values(data) as SiteNameProps[]).map(
          (value: SiteNameProps) => (
            <li key={value.name} className="flex py-1">
              <SiteCard
                title={value.name}
                imageSrc={`/images/${value.name.replace(/\s/g, "_")}.jpg`}
                alt={value.name}
                isFood={value.metadata.type === "food"}
              />
            </li>
          )
        )}
      </ul>
    </>
  );
};

export default Site;
