interface HeaderProps {
  className?: string;
}

export function Header({ className = "" }: HeaderProps) {
  return (
    <header className={`w-full bg-transparent px-6 py-4 ${className}`}>
      <div className="flex items-center justify-center">
        <img
          draggable={false}
          src="/assets/images/burnme.gif"
          alt="Burn Me While I'm Hot"
          className="h-8 object-contain"
        />
      </div>
    </header>
  );
}
