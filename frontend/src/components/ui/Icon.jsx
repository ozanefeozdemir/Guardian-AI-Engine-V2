import * as Lucide from 'lucide-react';

/**
 * Tasarım prototipi ikonları kebab-case isimle çağırıyor (örn. <Icon name="shield-alert" />).
 * Bu wrapper, ismi lucide-react'in PascalCase bileşen adına çevirip render eder.
 * Böylece tüm prototip çağrı noktaları değişmeden çalışır.
 */
const toPascal = (n) =>
  n.split('-').map((s) => s.charAt(0).toUpperCase() + s.slice(1)).join('');

export default function Icon({ name, size = 16, color, className = '', strokeWidth = 2, style }) {
  if (!name) return null;
  const Cmp = Lucide[toPascal(name)];
  if (!Cmp) {
    if (import.meta.env.DEV) console.warn('[Guardian] eksik lucide ikon:', name);
    return null;
  }
  return <Cmp size={size} color={color} className={className} strokeWidth={strokeWidth} style={style} />;
}
